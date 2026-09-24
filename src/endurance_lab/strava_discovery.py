from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db
from endurance_lab.strava_manifest import ManifestRow, load_manifest, save_manifest


DISCOVERY_URL = "https://www.strava.com/athlete/training_activities"


class DiscoveryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DiscoveredActivity:
    strava_id: str
    started_at: datetime
    sport: str
    name: str
    duration_seconds: float | None
    distance_m: float | None
    elevation_m: float | None


@dataclass
class DiscoverySummary:
    requested_from: date | None
    requested_to: date
    pages_checked: int = 0
    total_pages: int | None = None
    found: int = 0
    new: int = 0
    updated: int = 0
    existing: int = 0
    complete: bool = False
    activities: list[DiscoveredActivity] = field(default_factory=list)


class StravaDiscovery:
    """Incremental, paginated discovery through Strava's authenticated activity table."""

    def __init__(
        self,
        transport,
        *,
        delay_seconds: float = 1.0,
        max_retries: int = 2,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport
        self.delay_seconds = max(0.0, delay_seconds)
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.timeout_seconds = timeout_seconds
        self.sleep = sleep

    def discover(
        self,
        *,
        date_from: date | None,
        date_to: date,
        manifest_path: str | Path,
        database: str | Path | None = None,
        sync_run_id: int | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> DiscoverySummary:
        if date_from and date_to < date_from:
            raise ValueError("Discovery end date must not precede its start date")
        init_db(database)
        start_page = 1
        discovery_run_id = None
        if not dry_run:
            discovery_run_id, start_page = _start_or_resume(
                date_from, date_to, database, sync_run_id, force
            )
        rows = load_manifest(manifest_path)
        by_id = {row.strava_id: row for row in rows}
        summary = DiscoverySummary(date_from, date_to)
        page = start_page
        seen: set[str] = set()
        try:
            while True:
                payload = self._page(page)
                models = payload.get("models")
                if not isinstance(models, list):
                    raise DiscoveryError("DISCOVERY_ERROR", "Strava discovery response has no activity list")
                per_page = int(payload.get("perPage") or len(models) or 20)
                total = int(payload.get("total") or 0)
                summary.total_pages = math.ceil(total / max(per_page, 1))
                summary.pages_checked += 1
                oldest: date | None = None
                for model in models:
                    activity = _parse_activity(model)
                    activity_day = activity.started_at.date()
                    oldest = min(oldest, activity_day) if oldest else activity_day
                    if activity_day > date_to or (date_from and activity_day < date_from):
                        continue
                    if activity.strava_id in seen:
                        continue
                    seen.add(activity.strava_id)
                    summary.activities.append(activity)
                    summary.found += 1
                    current = by_id.get(activity.strava_id)
                    if current is None:
                        current = ManifestRow(activity.strava_id)
                        rows.append(current)
                        by_id[activity.strava_id] = current
                        summary.new += 1
                    else:
                        changed = _metadata_changed(current, activity)
                        summary.updated += int(changed)
                        summary.existing += int(not changed)
                    if not dry_run:
                        _apply_metadata(current, activity)
                if not dry_run:
                    save_manifest(rows, manifest_path)
                    _checkpoint(discovery_run_id, page, summary, database)
                exhausted = not models or page >= (summary.total_pages or page)
                crossed_start = bool(date_from and oldest and oldest < date_from)
                if exhausted or crossed_start:
                    summary.complete = True
                    break
                page += 1
                if self.delay_seconds:
                    self.sleep(self.delay_seconds)
            if not dry_run:
                _finish(discovery_run_id, page, summary, database)
            return summary
        except Exception as exc:
            if not dry_run and discovery_run_id is not None:
                _fail(discovery_run_id, str(exc), database)
            raise

    def _page(self, page: int) -> dict[str, Any]:
        query = urlencode({"new_activity_only": "false", "page": page, "per_page": 20})
        for retry in range(self.max_retries + 1):
            try:
                response = self.transport.get(
                    f"{DISCOVERY_URL}?{query}",
                    timeout=self.timeout_seconds,
                    headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
                )
            except Exception as exc:
                if retry < self.max_retries:
                    self.sleep(self.backoff_seconds * (2**retry))
                    continue
                raise DiscoveryError("NETWORK_ERROR", str(exc)) from exc
            if response.status_code in {401, 403}:
                raise DiscoveryError("AUTH_REQUIRED", f"Strava discovery returned HTTP {response.status_code}")
            if response.status_code == 429:
                raise DiscoveryError("RATE_LIMITED", "Strava discovery returned HTTP 429")
            if response.status_code in {408, 425, 500, 502, 503, 504} and retry < self.max_retries:
                self.sleep(self.backoff_seconds * (2**retry))
                continue
            if response.status_code != 200:
                raise DiscoveryError("DISCOVERY_ERROR", f"Strava discovery returned HTTP {response.status_code}")
            try:
                value = json.loads(response.content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DiscoveryError("DISCOVERY_ERROR", "Strava discovery did not return JSON") from exc
            if not isinstance(value, dict):
                raise DiscoveryError("DISCOVERY_ERROR", "Unexpected Strava discovery response")
            return value
        raise DiscoveryError("NETWORK_ERROR", "Strava discovery retry limit reached")


def default_discovery_window(
    manifest_path: str | Path,
    database: str | Path | None = None,
    *,
    today: date | None = None,
) -> tuple[date, date]:
    today = today or date.today()
    config = load_athlete_config()
    overlap = int(config.get("strava_sync", {}).get("lookback_days", 5))
    candidates: list[date] = []
    for row in load_manifest(manifest_path):
        parsed = _parse_date(row.activity_date)
        if parsed:
            candidates.append(parsed)
    init_db(database)
    with connect(database) as connection:
        row = connection.execute(
            "SELECT MAX(substr(started_at, 1, 10)) FROM activities WHERE source_activity_id IS NOT NULL"
        ).fetchone()
        parsed = _parse_date(row[0] if row else None)
        if parsed:
            candidates.append(parsed)
    latest = max(candidates) if candidates else today - timedelta(days=overlap)
    return latest - timedelta(days=overlap), today


def merge_discovered(
    activities: list[DiscoveredActivity], manifest_path: str | Path
) -> tuple[int, int, int]:
    rows = load_manifest(manifest_path)
    by_id = {row.strava_id: row for row in rows}
    new = updated = existing = 0
    for activity in activities:
        row = by_id.get(activity.strava_id)
        if row is None:
            row = ManifestRow(activity.strava_id)
            rows.append(row)
            by_id[activity.strava_id] = row
            new += 1
        elif _metadata_changed(row, activity):
            updated += 1
        else:
            existing += 1
        _apply_metadata(row, activity)
    save_manifest(rows, manifest_path)
    return new, updated, existing


def _parse_activity(model: dict[str, Any]) -> DiscoveredActivity:
    activity_id = str(model.get("id_str") or model.get("id") or "").strip()
    if not activity_id.isdigit():
        raise DiscoveryError("DISCOVERY_ERROR", "Discovered activity has no valid Strava ID")
    raw_time = str(model.get("start_time") or "").strip()
    try:
        started = datetime.strptime(raw_time, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError as exc:
        raise DiscoveryError("DISCOVERY_ERROR", f"Activity {activity_id} has an invalid start time") from exc
    return DiscoveredActivity(
        activity_id,
        started,
        str(model.get("sport_type") or model.get("display_type") or "Other"),
        str(model.get("name") or activity_id),
        _number(model.get("moving_time_raw") or model.get("elapsed_time_raw")),
        _number(model.get("distance_raw")),
        _number(model.get("elevation_gain_raw")),
    )


def _apply_metadata(row: ManifestRow, activity: DiscoveredActivity) -> None:
    now = datetime.now(timezone.utc).isoformat()
    row.activity_date = activity.started_at.isoformat()
    row.sport = activity.sport
    row.name = activity.name
    row.duration_seconds = _text_number(activity.duration_seconds)
    row.distance_m = _text_number(activity.distance_m)
    row.elevation_m = _text_number(activity.elevation_m)
    row.discovered_at = row.discovered_at or now
    row.discovery_source = "strava_training_activities"
    row.last_checked = now


def _metadata_changed(row: ManifestRow, activity: DiscoveredActivity) -> bool:
    return any((
        row.activity_date != activity.started_at.isoformat(),
        row.sport != activity.sport,
        row.name != activity.name,
        row.duration_seconds != _text_number(activity.duration_seconds),
        row.distance_m != _text_number(activity.distance_m),
    ))


def _start_or_resume(date_from, date_to, database, sync_run_id, force) -> tuple[int, int]:
    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        existing = None if force else connection.execute(
            """SELECT * FROM strava_discovery_runs
               WHERE requested_from IS ? AND requested_to=? AND status IN ('running','interrupted','failed')
               ORDER BY id DESC LIMIT 1""",
            (date_from.isoformat() if date_from else None, date_to.isoformat()),
        ).fetchone()
        if existing:
            connection.execute(
                "UPDATE strava_discovery_runs SET status='running', error_summary=NULL WHERE id=?",
                (existing["id"],),
            )
            connection.commit()
            return int(existing["id"]), int(existing["current_page"] or 0) + 1
        cursor = connection.execute(
            """INSERT INTO strava_discovery_runs
               (sync_run_id, requested_from, requested_to, status, started_at)
               VALUES (?, ?, ?, 'running', ?)""",
            (sync_run_id, date_from.isoformat() if date_from else None, date_to.isoformat(), now),
        )
        connection.commit()
        return int(cursor.lastrowid), 1


def _checkpoint(run_id, page, summary, database) -> None:
    with connect(database) as connection:
        connection.execute(
            "UPDATE strava_discovery_runs SET current_page=?, total_pages=?, discovered_count=? WHERE id=?",
            (page, summary.total_pages, summary.found, run_id),
        )
        connection.commit()


def _finish(run_id, page, summary, database) -> None:
    with connect(database) as connection:
        connection.execute(
            """UPDATE strava_discovery_runs SET status='success', finished_at=?, current_page=?,
                      total_pages=?, discovered_count=? WHERE id=?""",
            (datetime.now(timezone.utc).isoformat(), page,
             summary.total_pages, summary.found, run_id),
        )
        connection.commit()


def _fail(run_id, message, database) -> None:
    with connect(database) as connection:
        connection.execute(
            "UPDATE strava_discovery_runs SET status='failed', finished_at=?, error_summary=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), message[:1000], run_id),
        )
        connection.commit()


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _text_number(value: float | None) -> str:
    return "" if value is None else f"{value:g}"
