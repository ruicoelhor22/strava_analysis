from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from endurance_lab.db import connect, init_db


def ingestion_quality_report(database: str | Path | None = None) -> dict[str, Any]:
    init_db(database)
    with connect(database) as connection:
        activities = connection.execute("SELECT * FROM activities ORDER BY started_at").fetchall()
        by_sport = Counter(str(row["sport"]) for row in activities)
        by_format = Counter(str(row["source_format"]) for row in activities)
        coverage = Counter()
        suspicious: list[dict[str, Any]] = []
        for row in activities:
            activity_id = int(row["id"])
            stream = connection.execute(
                """SELECT COUNT(*) AS points,
                          SUM(heart_rate IS NOT NULL) AS hr_points,
                          SUM(latitude IS NOT NULL AND longitude IS NOT NULL) AS gps_points,
                          SUM(altitude_m IS NOT NULL) AS elevation_points,
                          SUM(cadence IS NOT NULL) AS cadence_points,
                          SUM(power_w IS NOT NULL) AS power_points,
                          MIN(recorded_at) AS first_time, MAX(recorded_at) AS last_time,
                          MAX(heart_rate) AS stream_max_hr, MAX(power_w) AS stream_max_power
                   FROM activity_streams WHERE activity_id = ?""",
                (activity_id,),
            ).fetchone()
            lap_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM activity_laps WHERE activity_id = ?", (activity_id,)
                ).fetchone()[0]
            )
            if row["avg_hr"] is not None or stream["hr_points"]:
                coverage["heart_rate"] += 1
            if stream["gps_points"]:
                coverage["gps"] += 1
            if row["ascent_m"] is not None or stream["elevation_points"]:
                coverage["elevation"] += 1
            if row["avg_cadence"] is not None or stream["cadence_points"]:
                coverage["cadence"] += 1
            if row["avg_power_w"] is not None or stream["power_points"]:
                coverage["power"] += 1
            if lap_count:
                coverage["laps"] += 1
            if row["distance_m"] is not None and row["distance_m"] > 0:
                coverage["distance"] += 1

            flags = _flags(connection, row, stream)
            if flags:
                suspicious.append(
                    {
                        "activity_id": activity_id,
                        "started_at": row["started_at"],
                        "sport": row["sport"],
                        "source_filename": row["source_filename"],
                        "flags": flags,
                    }
                )
        import_rows = connection.execute(
            "SELECT source_format, status, COUNT(*) AS count FROM imports GROUP BY source_format, status"
        ).fetchall()
    return {
        "total_activities": len(activities),
        "activities_by_sport": dict(sorted(by_sport.items())),
        "activities_by_source_format": dict(sorted(by_format.items())),
        "activities_containing": {
            key: int(coverage.get(key, 0))
            for key in ("heart_rate", "gps", "elevation", "cadence", "power", "laps", "distance")
        },
        "imports": [dict(row) for row in import_rows],
        "suspicious_activities": suspicious,
    }


def _flags(connection, activity, stream) -> list[str]:
    flags: list[str] = []
    duration = float(activity["elapsed_seconds"] or 0)
    distance = float(activity["distance_m"] or 0)
    sport = str(activity["sport"])
    points = int(stream["points"] or 0)
    max_hr = max(float(activity["max_hr"] or 0), float(stream["stream_max_hr"] or 0))
    max_power = max(float(activity["max_power_w"] or 0), float(stream["stream_max_power"] or 0))
    if duration <= 0:
        flags.append("zero_duration")
    if not activity["started_at"]:
        flags.append("missing_timestamp")
    if bool(activity["metadata_only"]):
        return flags
    if sport in {"cycling", "running", "swimming"} and distance <= 0:
        flags.append("zero_distance_unexpected")
    if max_hr > 240 or (max_hr and max_hr < 30):
        flags.append("impossible_heart_rate")
    if max_power > 3000:
        flags.append("impossible_power")
    if sport in {"cycling", "running", "swimming"} and duration > 600:
        if points == 0:
            flags.append("missing_stream")
        elif points < duration / 30:
            flags.append("unexpectedly_sparse_trackpoints")
    if points > 1 and stream["first_time"] and stream["last_time"]:
        first = datetime.fromisoformat(stream["first_time"])
        last = datetime.fromisoformat(stream["last_time"])
        span = max(0.0, (last - first).total_seconds())
        if span > 0 and points / span < 0.02:
            flags.append("corrupt_or_truncated_stream")
    gps_gaps = connection.execute(
        """WITH gps AS (
               SELECT recorded_at,
                      LAG(recorded_at) OVER (ORDER BY sequence) AS previous_at
               FROM activity_streams
               WHERE activity_id = ? AND latitude IS NOT NULL AND longitude IS NOT NULL
           )
           SELECT COUNT(*) FROM gps
           WHERE previous_at IS NOT NULL
             AND (julianday(recorded_at) - julianday(previous_at)) * 86400.0 > 300""",
        (activity["id"],),
    ).fetchone()[0]
    if gps_gaps:
        flags.append("large_gps_time_gap")
    return flags
