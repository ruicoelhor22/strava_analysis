from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
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
        issues: list[dict[str, Any]] = []
        severity_counts: Counter[str] = Counter()
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
                          MIN(heart_rate) AS stream_min_hr, MAX(heart_rate) AS stream_max_hr,
                          MAX(power_w) AS stream_max_power, MAX(speed_mps) AS stream_max_speed,
                          MAX(cadence) AS stream_max_cadence
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

            activity_issues = _issues(connection, row, stream)
            for issue in activity_issues:
                severity_counts[issue["severity"]] += 1
                issues.append({"activity_id": activity_id, **issue})
            important = [issue for issue in activity_issues if issue["severity"] in {"warning", "error"}]
            if important:
                suspicious.append(
                    {
                        "activity_id": activity_id,
                        "started_at": row["started_at"],
                        "sport": row["sport"],
                        "source_filename": row["source_filename"],
                        "flags": [issue["code"] for issue in important],
                        "severity": "error" if any(issue["severity"] == "error" for issue in important) else "warning",
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
        "issues": issues,
        "severity_counts": {level: int(severity_counts[level]) for level in ("error", "warning", "advisory", "info")},
    }


def _issues(connection, activity, stream) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    def add(code: str, severity: str, message: str) -> None:
        issues.append({"code": code, "severity": severity, "message": message})

    duration = float(activity["elapsed_seconds"] or 0)
    distance = float(activity["distance_m"] or 0)
    sport = str(activity["sport"])
    points = int(stream["points"] or 0)
    max_hr = max(float(activity["max_hr"] or 0), float(stream["stream_max_hr"] or 0))
    max_power = max(float(activity["max_power_w"] or 0), float(stream["stream_max_power"] or 0))
    if duration <= 0:
        add("zero_duration", "error", "Activity has no positive duration.")
    if not activity["started_at"]:
        add("missing_timestamp", "error", "Activity has no start timestamp.")
    if bool(activity["metadata_only"]):
        add("metadata_only", "info", "No export stream is available; summary fields remain usable.")
        return issues
    if sport in {"cycling", "running", "swimming"} and distance <= 0 and duration > 600:
        add("distance_unavailable", "advisory", "Distance is unavailable; this may be an indoor session.")
    min_hr = float(stream["stream_min_hr"] or 0)
    if max_hr > 240 or (min_hr and min_hr < 30):
        add("implausible_heart_rate", "warning", "Recorded heart rate falls outside a broad plausible range (30-240 bpm).")
    if max_power > 3000:
        add("implausible_power", "warning", "Recorded power exceeds 3000 W.")
    if float(stream["stream_max_speed"] or 0) > 45:
        add("implausible_speed", "warning", "Recorded speed exceeds 162 km/h.")
    if float(stream["stream_max_cadence"] or 0) > 250:
        add("implausible_cadence", "warning", "Recorded cadence exceeds 250 rpm.")
    if sport in {"cycling", "running", "swimming"} and duration > 600:
        if points == 0:
            add("missing_stream", "advisory", "No trackpoints were present for this endurance activity.")
        elif points < duration / 30:
            add("sparse_trackpoints", "advisory", "Recorded trackpoints are sparse relative to duration.")
    if points > 1 and stream["first_time"] and stream["last_time"]:
        first = datetime.fromisoformat(stream["first_time"])
        last = datetime.fromisoformat(stream["last_time"])
        span = max(0.0, (last - first).total_seconds())
        if span > 0 and points / span < 0.02:
            add("truncated_stream", "advisory", "Stream covers the session sparsely; check the source if detailed metrics matter.")
    gps = connection.execute(
        """SELECT recorded_at,latitude,longitude FROM activity_streams
           WHERE activity_id=? AND latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY sequence""",
        (activity["id"],),
    ).fetchall()
    large_gaps: list[float] = []
    impossible_jump = False
    for previous, current in zip(gps, gps[1:]):
        try:
            seconds = (datetime.fromisoformat(current["recorded_at"]) - datetime.fromisoformat(previous["recorded_at"])).total_seconds()
        except (TypeError, ValueError):
            continue
        if seconds > 300:
            large_gaps.append(seconds)
        if 0 < seconds <= 10:
            metres = _haversine(previous["latitude"], previous["longitude"], current["latitude"], current["longitude"])
            if metres > 1000 and metres / seconds > 60:
                impossible_jump = True
    if impossible_jump:
        add("impossible_gps_jump", "warning", "Consecutive GPS points imply an impossible jump.")
    if large_gaps:
        pause_budget = max(0.0, duration - float(activity["moving_seconds"] or duration))
        largest = max(large_gaps)
        if pause_budget >= 0.7 * largest:
            add("recording_pause", "info", f"A {largest / 60:.0f}-minute GPS gap is consistent with elapsed-minus-moving time.")
        elif largest >= 1200 and sum(large_gaps) >= duration * 0.1:
            add("substantial_gps_gap", "warning", f"GPS is absent for at least {largest / 60:.0f} consecutive minutes without a matching pause.")
        else:
            add("gps_gap", "advisory", "GPS has a gap not fully explained by moving-time pauses.")
    return issues


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lat2 = radians(lat1), radians(lat2)
    dlat, dlon = lat2 - lat1, radians(lon2 - lon1)
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6_371_000 * asin(min(1.0, sqrt(value)))
