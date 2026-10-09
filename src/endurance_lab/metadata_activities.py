from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from endurance_lab.db import init_db, transaction
from endurance_lab.plan_parser import normalize_plan_sport


@dataclass(frozen=True)
class MetadataActivity:
    source_activity_id: str
    started_at: datetime
    sport: str
    name: str | None = None
    duration_seconds: float | None = None
    distance_m: float | None = None
    source: str = "metadata_import"
    source_reference: str = "metadata"


def upsert_metadata_activity(
    activity: MetadataActivity,
    database: str | Path | None = None,
) -> int:
    """Store a summary-only activity without pretending stream data exists."""
    if not str(activity.source_activity_id).strip():
        raise ValueError("Metadata activities require a source activity ID")
    init_db(database)
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "source_activity_id": str(activity.source_activity_id),
        "started_at": activity.started_at.isoformat(),
        "sport": normalize_plan_sport(activity.sport),
        "name": activity.name,
        "duration_seconds": activity.duration_seconds,
        "distance_m": activity.distance_m,
        "source": activity.source,
        "source_reference": activity.source_reference,
        "metadata_only": True,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    filename = f"metadata-{activity.source_activity_id}.json"
    stable_id = f"metadata:{activity.source}:{activity.source_activity_id}"
    with transaction(database) as connection:
        existing = connection.execute(
            "SELECT id, metadata_only FROM activities WHERE source_activity_id = ?",
            (str(activity.source_activity_id),),
        ).fetchone()
        if existing and not int(existing["metadata_only"]):
            return int(existing["id"])
        candidate_id = None if existing else _matching_detailed_candidate(connection, activity)
        connection.execute(
            """INSERT OR IGNORE INTO imports (
                   source_filename, sha256, source_format, source, raw_path,
                   imported_at, status, activity_count
               ) VALUES (?, ?, 'metadata', ?, ?, ?, 'imported', 1)""",
            (filename, digest, activity.source, activity.source_reference, now),
        )
        import_id = int(connection.execute(
            "SELECT id FROM imports WHERE sha256 = ?", (digest,)
        ).fetchone()[0])
        if candidate_id is not None:
            connection.execute(
                """UPDATE activities SET source_activity_id=?, stable_id=?, identity_key=?
                   WHERE id=? AND source_activity_id IS NULL AND metadata_only=0""",
                (str(activity.source_activity_id), f"strava:{activity.source_activity_id}",
                 f"strava:{activity.source_activity_id}", candidate_id),
            )
            connection.execute(
                """INSERT OR IGNORE INTO activity_sources (
                       activity_id, import_id, source_format, source_filename, raw_path, sha256,
                       source_activity_id, quality_score, selected, metadata_json, imported_at
                   ) VALUES (?, ?, 'metadata', ?, ?, ?, ?, 1, 0, ?, ?)""",
                (candidate_id, import_id, filename, activity.source_reference, digest,
                 str(activity.source_activity_id), json.dumps(payload, ensure_ascii=False), now),
            )
            return candidate_id
        if existing:
            activity_id = int(existing["id"])
            connection.execute(
                """UPDATE activities SET name = ?, started_at = ?, elapsed_seconds = ?,
                          moving_seconds = ?, distance_m = ?, sport = ?, source_metadata_json = ?,
                          imported_at = ? WHERE id = ? AND metadata_only = 1""",
                (activity.name, activity.started_at.isoformat(), activity.duration_seconds,
                 activity.duration_seconds, activity.distance_m, normalize_plan_sport(activity.sport),
                 json.dumps(payload, ensure_ascii=False), now, activity_id),
            )
        else:
            cursor = connection.execute(
                """INSERT INTO activities (
                       stable_id, source_activity_id, source_filename, raw_path, import_id,
                       source_format, source, identity_key, source_metadata_json, quality_score,
                       metadata_only, sport, name, started_at, elapsed_seconds, moving_seconds,
                       distance_m, imported_at
                   ) VALUES (?, ?, ?, ?, ?, 'metadata', ?, ?, ?, 1, 1, ?, ?, ?, ?, ?, ?, ?)""",
                (stable_id, str(activity.source_activity_id), filename, activity.source_reference,
                 import_id, activity.source, stable_id, json.dumps(payload, ensure_ascii=False),
                 normalize_plan_sport(activity.sport), activity.name, activity.started_at.isoformat(),
                 activity.duration_seconds, activity.duration_seconds, activity.distance_m, now),
            )
            activity_id = int(cursor.lastrowid)
        connection.execute(
            """INSERT OR IGNORE INTO activity_sources (
                   activity_id, import_id, source_format, source_filename, raw_path, sha256,
                   source_activity_id, quality_score, selected, metadata_json, imported_at
               ) VALUES (?, ?, 'metadata', ?, ?, ?, ?, 1, 1, ?, ?)""",
            (activity_id, import_id, filename, activity.source_reference, digest,
             str(activity.source_activity_id), json.dumps(payload, ensure_ascii=False), now),
        )
    return activity_id


def _matching_detailed_candidate(connection, activity: MetadataActivity) -> int | None:
    if activity.duration_seconds is None or activity.duration_seconds <= 0:
        return None
    sport = normalize_plan_sport(activity.sport)
    matches = connection.execute(
        """SELECT id,elapsed_seconds,distance_m FROM activities
           WHERE source_activity_id IS NULL AND metadata_only=0 AND sport=?
             AND ABS((julianday(started_at)-julianday(?))*86400.0) <= 5""",
        (sport, activity.started_at.isoformat()),
    ).fetchall()
    eligible = []
    for row in matches:
        elapsed = row["elapsed_seconds"]
        if elapsed is None or abs(float(elapsed) - activity.duration_seconds) > max(10, activity.duration_seconds * 0.02):
            continue
        local_distance = row["distance_m"]
        if activity.distance_m is not None and local_distance is not None:
            if abs(float(local_distance) - activity.distance_m) > max(100, activity.distance_m * 0.02):
                continue
        eligible.append(int(row["id"]))
    return eligible[0] if len(eligible) == 1 else None
