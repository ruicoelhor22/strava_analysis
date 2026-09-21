from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from endurance_lab.config import ensure_local_layout, paths
from endurance_lab.db import connect, init_db
from endurance_lab.fit import parse_fit
from endurance_lab.identity import deterministic_identity
from endurance_lab.models import Lap, ParsedActivity, StrengthSet, Trackpoint
from endurance_lab.normalize import activity_quality, finalize_activity
from endurance_lab.strength_json import parse_strength_json
from endurance_lab.tcx import parse_tcx


SUPPORTED_FORMATS = {".tcx": "tcx", ".fit": "fit", ".json": "json"}
PARSERS: dict[str, Callable[[str | Path, str | None], list[ParsedActivity]]] = {
    "tcx": parse_tcx,
    "fit": parse_fit,
    "json": parse_strength_json,
}


@dataclass(frozen=True)
class ImportResult:
    file: Path
    source_format: str
    status: str
    imported_activities: int = 0
    upgraded_activities: int = 0
    skipped_activities: int = 0
    message: str | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files(source: str | Path | None = None) -> list[Path]:
    target = Path(source) if source else ensure_local_layout().import_dir
    if not target.exists():
        raise FileNotFoundError(target)
    return [target] if target.is_file() else sorted(path for path in target.rglob("*") if path.is_file())


def import_path(
    source: str | Path | None = None, database: str | Path | None = None
) -> list[ImportResult]:
    ensure_local_layout()
    return [import_file(file, database=database) for file in discover_files(source)]


def import_file(source: str | Path, database: str | Path | None = None) -> ImportResult:
    source_path = Path(source).resolve()
    if not source_path.is_file():
        return ImportResult(source_path, "unknown", "failed", message="File not found")
    source_format = SUPPORTED_FORMATS.get(source_path.suffix.lower())
    if source_format is None:
        return ImportResult(source_path, source_path.suffix.lower().lstrip(".") or "unknown", "unsupported")

    init_db(database)
    digest = sha256_file(source_path)
    layout = paths()
    raw_path = layout.raw_dir / source_format / f"{digest[:16]}_{_safe_name(source_path.name)}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with connect(database) as connection:
        existing_import = connection.execute(
            "SELECT id, status, activity_count FROM imports WHERE sha256 = ?", (digest,)
        ).fetchone()
        if existing_import and existing_import["status"] == "imported":
            return ImportResult(
                source_path,
                source_format,
                "duplicate",
                skipped_activities=int(existing_import["activity_count"] or 0),
                message="Exact file content already imported",
            )
        if existing_import:
            connection.execute("DELETE FROM imports WHERE id = ?", (existing_import["id"],))

    if not raw_path.exists():
        shutil.copy2(source_path, raw_path)
    imported_at = datetime.now(timezone.utc).isoformat()
    try:
        activities = PARSERS[source_format](source_path, digest)
    except Exception as exc:
        _record_failed_import(
            database, source_path.name, digest, raw_path, source_format, imported_at, exc
        )
        return ImportResult(source_path, source_format, "failed", message=str(exc))

    imported = upgraded = skipped = 0
    try:
        with connect(database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """INSERT INTO imports (
                       source_filename, sha256, source_format, source, raw_path,
                       imported_at, status, activity_count
                   ) VALUES (?, ?, ?, 'manual_file', ?, ?, 'processing', ?)""",
                (source_path.name, digest, source_format, str(raw_path), imported_at, len(activities)),
            )
            import_id = int(cursor.lastrowid)
            for parsed in activities:
                parsed = finalize_activity(parsed)
                match = _find_match(connection, parsed)
                if match is None:
                    activity_id = _insert_activity(
                        connection,
                        parsed,
                        source_path.name,
                        raw_path,
                        source_format,
                        import_id,
                        imported_at,
                    )
                    _insert_source(
                        connection, activity_id, parsed, import_id, source_path.name,
                        raw_path, digest, source_format, imported_at, selected=True,
                    )
                    imported += 1
                    continue

                activity_id = int(match["id"])
                existing = _load_activity(connection, activity_id)
                merged, enriched, select_new = _merge_activities(existing, parsed)
                if enriched:
                    _replace_activity(
                        connection,
                        activity_id,
                        merged,
                        source_path.name,
                        raw_path,
                        source_format,
                        import_id,
                        imported_at,
                        select_new,
                    )
                    upgraded += 1
                else:
                    score, _ = activity_quality(existing)
                    connection.execute(
                        "UPDATE activities SET quality_score = MAX(quality_score, ?) WHERE id = ?",
                        (score, activity_id),
                    )
                    skipped += 1
                _insert_source(
                    connection, activity_id, parsed, import_id, source_path.name,
                    raw_path, digest, source_format, imported_at, selected=select_new,
                )
                if select_new:
                    connection.execute(
                        "UPDATE activity_sources SET selected = CASE WHEN import_id = ? THEN 1 ELSE 0 END WHERE activity_id = ?",
                        (import_id, activity_id),
                    )
            connection.execute("UPDATE imports SET status = 'imported' WHERE id = ?", (import_id,))
            connection.commit()
    except Exception as exc:
        _record_failed_import(
            database, source_path.name, digest, raw_path, source_format, imported_at, exc
        )
        return ImportResult(source_path, source_format, "failed", message=str(exc))

    status = "upgraded" if upgraded else "imported" if imported else "skipped"
    return ImportResult(source_path, source_format, status, imported, upgraded, skipped)


def _find_match(connection: sqlite3.Connection, activity: ParsedActivity):
    if activity.source_activity_id:
        row = connection.execute(
            "SELECT * FROM activities WHERE source_activity_id = ? LIMIT 1",
            (activity.source_activity_id,),
        ).fetchone()
        if row:
            return row
    if activity.stable_id.startswith("strava:"):
        row = connection.execute(
            "SELECT * FROM activities WHERE stable_id = ? LIMIT 1", (activity.stable_id,)
        ).fetchone()
        if row:
            return row
    candidates = connection.execute(
        """SELECT * FROM activities
           WHERE ABS((julianday(started_at) - julianday(?)) * 86400.0) <= 5.0""",
        (activity.started_at.isoformat(),),
    ).fetchall()
    for row in candidates:
        if not _sports_compatible(str(row["sport"]), activity.sport):
            continue
        if not _near(row["elapsed_seconds"], activity.elapsed_seconds, floor=10, ratio=0.02):
            continue
        if not _near(row["distance_m"], activity.distance_m, floor=100, ratio=0.02):
            continue
        return row
    return None


def _insert_activity(
    connection: sqlite3.Connection,
    activity: ParsedActivity,
    filename: str,
    raw_path: Path,
    source_format: str,
    import_id: int,
    imported_at: str,
) -> int:
    score, _ = activity_quality(activity)
    identity = (
        f"strava:{activity.source_activity_id}"
        if activity.source_activity_id
        else deterministic_identity(activity.started_at, activity.sport)
    )
    cursor = connection.execute(
        """INSERT INTO activities (
            stable_id, source_activity_id, source_filename, raw_path, import_id,
            source_format, source, identity_key, source_metadata_json, quality_score,
            sport, name, started_at, ended_at, elapsed_seconds, moving_seconds,
            distance_m, ascent_m, descent_m, calories, avg_hr, max_hr,
            avg_speed_mps, max_speed_mps, avg_cadence, max_cadence,
            avg_power_w, max_power_w, device, imported_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, 'manual_file', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )""",
        _activity_values(
            activity, filename, raw_path, source_format, import_id, imported_at,
            identity, score,
        ),
    )
    activity_id = int(cursor.lastrowid)
    _write_children(connection, activity_id, activity)
    return activity_id


def _replace_activity(
    connection: sqlite3.Connection,
    activity_id: int,
    activity: ParsedActivity,
    filename: str,
    raw_path: Path,
    source_format: str,
    import_id: int,
    imported_at: str,
    select_new: bool,
) -> None:
    score, _ = activity_quality(activity)
    identity = (
        f"strava:{activity.source_activity_id}"
        if activity.source_activity_id
        else deterministic_identity(activity.started_at, activity.sport)
    )
    connection.execute(
        """UPDATE activities SET
            stable_id = ?, source_activity_id = COALESCE(?, source_activity_id),
            identity_key = ?, quality_score = ?, sport = ?, name = ?, started_at = ?,
            ended_at = ?, elapsed_seconds = ?, moving_seconds = ?, distance_m = ?,
            ascent_m = ?, descent_m = ?, calories = ?, avg_hr = ?, max_hr = ?,
            avg_speed_mps = ?, max_speed_mps = ?, avg_cadence = ?, max_cadence = ?,
            avg_power_w = ?, max_power_w = ?, device = ?,
            source_filename = CASE WHEN ? THEN ? ELSE source_filename END,
            raw_path = CASE WHEN ? THEN ? ELSE raw_path END,
            import_id = CASE WHEN ? THEN ? ELSE import_id END,
            source_format = CASE WHEN ? THEN ? ELSE source_format END,
            source_metadata_json = CASE WHEN ? THEN ? ELSE source_metadata_json END,
            imported_at = CASE WHEN ? THEN ? ELSE imported_at END
           WHERE id = ?""",
        (
            activity.stable_id, activity.source_activity_id, identity, score, activity.sport,
            activity.name, activity.started_at.isoformat(), _iso(activity.ended_at),
            activity.elapsed_seconds, activity.moving_seconds, activity.distance_m,
            activity.ascent_m, activity.descent_m, activity.calories, activity.avg_hr,
            activity.max_hr, activity.avg_speed_mps, activity.max_speed_mps,
            activity.avg_cadence, activity.max_cadence, activity.avg_power_w,
            activity.max_power_w, activity.device,
            select_new, filename, select_new, str(raw_path), select_new, import_id,
            select_new, source_format, select_new,
            json.dumps(activity.source_metadata, default=str), select_new, imported_at,
            activity_id,
        ),
    )
    connection.execute("DELETE FROM activity_laps WHERE activity_id = ?", (activity_id,))
    connection.execute("DELETE FROM activity_streams WHERE activity_id = ?", (activity_id,))
    connection.execute("DELETE FROM strength_sets WHERE activity_id = ?", (activity_id,))
    _write_children(connection, activity_id, activity)


def _insert_source(
    connection, activity_id, activity, import_id, filename, raw_path, digest,
    source_format, imported_at, selected,
) -> None:
    score, coverage = activity_quality(activity)
    metadata = {**activity.source_metadata, "coverage": coverage}
    connection.execute(
        """INSERT OR IGNORE INTO activity_sources (
               activity_id, import_id, source_format, source_filename, raw_path,
               sha256, source_activity_id, quality_score, selected, metadata_json, imported_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            activity_id, import_id, source_format, filename, str(raw_path), digest,
            activity.source_activity_id, score, int(selected),
            json.dumps(metadata, default=str), imported_at,
        ),
    )


def _write_children(connection: sqlite3.Connection, activity_id: int, activity: ParsedActivity) -> None:
    connection.executemany(
        """INSERT INTO activity_laps (
            activity_id, lap_number, started_at, ended_at, duration_seconds,
            distance_m, ascent_m, descent_m, calories, avg_hr, max_hr,
            avg_speed_mps, max_speed_mps, avg_cadence, max_cadence,
            avg_power_w, max_power_w
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                activity_id, lap.lap_number, _iso(lap.started_at), _iso(lap.ended_at),
                lap.duration_seconds, lap.distance_m, lap.ascent_m, lap.descent_m,
                lap.calories, lap.avg_hr, lap.max_hr, lap.avg_speed_mps,
                lap.max_speed_mps, lap.avg_cadence, lap.max_cadence,
                lap.avg_power_w, lap.max_power_w,
            )
            for lap in activity.laps
        ],
    )
    connection.executemany(
        """INSERT INTO activity_streams (
            activity_id, sequence, recorded_at, elapsed_seconds, distance_m,
            latitude, longitude, altitude_m, heart_rate, cadence, speed_mps,
            power_w, temperature_c, moving
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                activity_id, point.sequence, _iso(point.recorded_at), point.elapsed_seconds,
                point.distance_m, point.latitude, point.longitude, point.altitude_m,
                point.heart_rate, point.cadence, point.speed_mps, point.power_w,
                point.temperature_c, None if point.moving is None else int(point.moving),
            )
            for point in activity.trackpoints
        ],
    )
    connection.executemany(
        """INSERT INTO strength_sets (
               activity_id, exercise_name, set_number, repetitions, load_value,
               load_unit, started_at, duration_seconds, metadata_json
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                activity_id, item.exercise_name, item.set_number, item.repetitions,
                item.load_value, item.load_unit, _iso(item.started_at),
                item.duration_seconds, json.dumps(item.metadata, default=str),
            )
            for item in activity.strength_sets
        ],
    )


def _load_activity(connection: sqlite3.Connection, activity_id: int) -> ParsedActivity:
    row = connection.execute("SELECT * FROM activities WHERE id = ?", (activity_id,)).fetchone()
    laps = [Lap(
        lap_number=item["lap_number"], started_at=_dt(item["started_at"]),
        ended_at=_dt(item["ended_at"]), duration_seconds=item["duration_seconds"],
        distance_m=item["distance_m"], ascent_m=item["ascent_m"], descent_m=item["descent_m"],
        calories=item["calories"], avg_hr=item["avg_hr"], max_hr=item["max_hr"],
        avg_speed_mps=item["avg_speed_mps"], max_speed_mps=item["max_speed_mps"],
        avg_cadence=item["avg_cadence"], max_cadence=item["max_cadence"],
        avg_power_w=item["avg_power_w"], max_power_w=item["max_power_w"],
    ) for item in connection.execute("SELECT * FROM activity_laps WHERE activity_id=? ORDER BY lap_number", (activity_id,))]
    points = [Trackpoint(
        sequence=item["sequence"], recorded_at=_dt(item["recorded_at"]),
        elapsed_seconds=item["elapsed_seconds"], distance_m=item["distance_m"],
        latitude=item["latitude"], longitude=item["longitude"], altitude_m=item["altitude_m"],
        heart_rate=item["heart_rate"], cadence=item["cadence"], speed_mps=item["speed_mps"],
        power_w=item["power_w"], temperature_c=item["temperature_c"],
        moving=None if item["moving"] is None else bool(item["moving"]),
    ) for item in connection.execute("SELECT * FROM activity_streams WHERE activity_id=? ORDER BY sequence", (activity_id,))]
    sets = [StrengthSet(
        exercise_name=item["exercise_name"], set_number=item["set_number"],
        repetitions=item["repetitions"], load_value=item["load_value"],
        load_unit=item["load_unit"], started_at=_dt(item["started_at"]),
        duration_seconds=item["duration_seconds"], metadata=json.loads(item["metadata_json"] or "{}"),
    ) for item in connection.execute("SELECT * FROM strength_sets WHERE activity_id=? ORDER BY id", (activity_id,))]
    return ParsedActivity(
        stable_id=row["stable_id"], source_activity_id=row["source_activity_id"],
        sport=row["sport"], started_at=_dt(row["started_at"]), ended_at=_dt(row["ended_at"]),
        name=row["name"], elapsed_seconds=row["elapsed_seconds"], moving_seconds=row["moving_seconds"],
        distance_m=row["distance_m"], ascent_m=row["ascent_m"], descent_m=row["descent_m"],
        calories=row["calories"], avg_hr=row["avg_hr"], max_hr=row["max_hr"],
        avg_speed_mps=row["avg_speed_mps"], max_speed_mps=row["max_speed_mps"],
        avg_cadence=row["avg_cadence"], max_cadence=row["max_cadence"],
        avg_power_w=row["avg_power_w"], max_power_w=row["max_power_w"], device=row["device"],
        source_metadata=json.loads(row["source_metadata_json"] or "{}"),
        laps=laps, trackpoints=points, strength_sets=sets,
    )


def _merge_activities(existing: ParsedActivity, incoming: ParsedActivity) -> tuple[ParsedActivity, bool, bool]:
    old_score, old_coverage = activity_quality(existing)
    new_score, _ = activity_quality(incoming)
    select_new = new_score > old_score
    prefer_new = select_new

    def pick(old, new, *, positive=False):
        new_valid = new is not None and (not positive or float(new) > 0)
        old_valid = old is not None and (not positive or float(old) > 0)
        if prefer_new and new_valid:
            return new
        if old_valid:
            return old
        return new if new_valid else old

    stable_id = existing.stable_id
    if incoming.source_activity_id and not existing.source_activity_id:
        stable_id = f"strava:{incoming.source_activity_id}"
    merged = ParsedActivity(
        stable_id=stable_id,
        source_activity_id=existing.source_activity_id or incoming.source_activity_id,
        sport=incoming.sport if existing.sport == "other" and incoming.sport != "other" else existing.sport,
        started_at=min(existing.started_at, incoming.started_at),
        ended_at=pick(existing.ended_at, incoming.ended_at),
        name=pick(existing.name, incoming.name),
        elapsed_seconds=pick(existing.elapsed_seconds, incoming.elapsed_seconds, positive=True),
        moving_seconds=pick(existing.moving_seconds, incoming.moving_seconds, positive=True),
        distance_m=pick(existing.distance_m, incoming.distance_m, positive=True),
        ascent_m=pick(existing.ascent_m, incoming.ascent_m),
        descent_m=pick(existing.descent_m, incoming.descent_m),
        calories=pick(existing.calories, incoming.calories),
        avg_hr=pick(existing.avg_hr, incoming.avg_hr, positive=True),
        max_hr=pick(existing.max_hr, incoming.max_hr, positive=True),
        avg_speed_mps=pick(existing.avg_speed_mps, incoming.avg_speed_mps, positive=True),
        max_speed_mps=pick(existing.max_speed_mps, incoming.max_speed_mps, positive=True),
        avg_cadence=pick(existing.avg_cadence, incoming.avg_cadence, positive=True),
        max_cadence=pick(existing.max_cadence, incoming.max_cadence, positive=True),
        avg_power_w=pick(existing.avg_power_w, incoming.avg_power_w, positive=True),
        max_power_w=pick(existing.max_power_w, incoming.max_power_w, positive=True),
        device=pick(existing.device, incoming.device),
        source_metadata={**existing.source_metadata, **incoming.source_metadata} if select_new else {**incoming.source_metadata, **existing.source_metadata},
        laps=incoming.laps if _lap_score(incoming.laps) > _lap_score(existing.laps) else existing.laps,
        trackpoints=_merge_trackpoints(existing.trackpoints, incoming.trackpoints, prefer_new),
        strength_sets=_merge_sets(existing.strength_sets, incoming.strength_sets, prefer_new),
    )
    finalize_activity(merged)
    merged_score, merged_coverage = activity_quality(merged)
    enriched = (
        merged_score > old_score
        or any(int(merged_coverage[key]) > int(old_coverage[key]) for key in merged_coverage if key in old_coverage)
        or _summary_completeness(merged) > _summary_completeness(existing)
    )
    return merged, enriched, select_new


def _merge_trackpoints(old: list[Trackpoint], new: list[Trackpoint], prefer_new: bool) -> list[Trackpoint]:
    combined: dict[tuple, Trackpoint] = {}
    for is_new, points in ((False, old), (True, new)):
        for point in points:
            key = (
                ("time", point.recorded_at.isoformat())
                if point.recorded_at
                else ("elapsed", round(point.elapsed_seconds or point.sequence, 3))
            )
            if key not in combined:
                combined[key] = point
                continue
            first, second = (
                (point, combined[key]) if prefer_new and is_new else (combined[key], point)
            )
            combined[key] = Trackpoint(
                sequence=0,
                recorded_at=first.recorded_at or second.recorded_at,
                elapsed_seconds=first.elapsed_seconds if first.elapsed_seconds is not None else second.elapsed_seconds,
                distance_m=first.distance_m if first.distance_m is not None else second.distance_m,
                latitude=first.latitude if first.latitude is not None else second.latitude,
                longitude=first.longitude if first.longitude is not None else second.longitude,
                altitude_m=first.altitude_m if first.altitude_m is not None else second.altitude_m,
                heart_rate=first.heart_rate if first.heart_rate is not None else second.heart_rate,
                cadence=first.cadence if first.cadence is not None else second.cadence,
                speed_mps=first.speed_mps if first.speed_mps is not None else second.speed_mps,
                power_w=first.power_w if first.power_w is not None else second.power_w,
                temperature_c=first.temperature_c if first.temperature_c is not None else second.temperature_c,
                moving=first.moving if first.moving is not None else second.moving,
            )
    return list(combined.values())


def _merge_sets(old, new, prefer_new):
    values = {}
    for item in (old + new if prefer_new else new + old):
        values[(item.exercise_name, item.set_number)] = item
    return list(values.values())


def _lap_score(laps: list[Lap]) -> int:
    return len(laps) * 2 + sum(lap.distance_m is not None for lap in laps) + sum(lap.avg_hr is not None for lap in laps)


def _summary_completeness(activity: ParsedActivity) -> int:
    fields = (
        activity.elapsed_seconds, activity.moving_seconds, activity.distance_m,
        activity.ascent_m, activity.avg_hr, activity.max_hr, activity.avg_speed_mps,
        activity.avg_cadence, activity.avg_power_w, activity.device,
    )
    return sum(value is not None for value in fields)


def _activity_values(activity, filename, raw_path, source_format, import_id, imported_at, identity, score):
    return (
        activity.stable_id, activity.source_activity_id, filename, str(raw_path), import_id,
        source_format, identity, json.dumps(activity.source_metadata, default=str), score,
        activity.sport, activity.name, activity.started_at.isoformat(), _iso(activity.ended_at),
        activity.elapsed_seconds, activity.moving_seconds, activity.distance_m,
        activity.ascent_m, activity.descent_m, activity.calories, activity.avg_hr,
        activity.max_hr, activity.avg_speed_mps, activity.max_speed_mps,
        activity.avg_cadence, activity.max_cadence, activity.avg_power_w,
        activity.max_power_w, activity.device, imported_at,
    )


def _record_failed_import(database, filename, digest, raw_path, source_format, imported_at, exc):
    with connect(database) as connection:
        connection.execute("DELETE FROM imports WHERE sha256 = ? AND status != 'imported'", (digest,))
        connection.execute(
            """INSERT OR IGNORE INTO imports (
                   source_filename, sha256, source_format, source, raw_path,
                   imported_at, status, error
               ) VALUES (?, ?, ?, 'manual_file', ?, ?, 'failed', ?)""",
            (filename, digest, source_format, str(raw_path), imported_at, str(exc)),
        )


def _sports_compatible(left: str, right: str) -> bool:
    return left == right or "other" in {left, right}


def _near(left, right, floor: float, ratio: float) -> bool:
    if left in (None, 0) or right in (None, 0):
        return True
    return abs(float(left) - float(right)) <= max(floor, max(float(left), float(right)) * ratio)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
