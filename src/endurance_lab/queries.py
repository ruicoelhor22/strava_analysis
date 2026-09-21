from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from endurance_lab.db import connect, init_db


ACTIVITY_SELECT = """
SELECT a.*, d.normalized_power_w, d.intensity_factor, d.estimated_tss,
       d.estimated_hr_load, d.selected_load, d.load_method,
       d.efficiency_factor, d.aerobic_decoupling_pct, d.decoupling_status,
       d.decoupling_reason, d.first_half_output, d.second_half_output,
       d.first_half_hr, d.second_half_hr, d.late_fade_pct,
       d.pace_seconds_per_km, d.hr_coverage, d.moving_ratio,
       d.hr_zones_json, d.power_zones_json, d.best_power_json, d.ftp_w
FROM activities a
LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
"""


def activities(
    database: str | Path | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    sports: Iterable[str] | None = None,
) -> pd.DataFrame:
    init_db(database)
    clauses: list[str] = []
    params: list[object] = []
    if start is not None:
        clauses.append("a.started_at >= ?")
        params.append(str(start))
    if end is not None:
        clauses.append("a.started_at < ?")
        params.append(str(end))
    sport_values = list(sports or [])
    if sport_values:
        clauses.append(f"a.sport IN ({','.join('?' for _ in sport_values)})")
        params.extend(sport_values)
    query = ACTIVITY_SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY a.started_at DESC"
    with connect(database) as connection:
        frame = pd.read_sql_query(query, connection, params=params)
    if not frame.empty:
        frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True, errors="coerce")
    return frame


def activity(activity_id: int, database: str | Path | None = None) -> dict | None:
    with connect(database) as connection:
        row = connection.execute(ACTIVITY_SELECT + " WHERE a.id = ?", (activity_id,)).fetchone()
    return dict(row) if row else None


def daily_load(database: str | Path | None = None) -> pd.DataFrame:
    init_db(database)
    with connect(database) as connection:
        frame = pd.read_sql_query("SELECT * FROM daily_training_load ORDER BY day", connection)
    if not frame.empty:
        frame["day"] = pd.to_datetime(frame["day"], errors="coerce")
    return frame


def activity_streams(
    activity_id: int, database: str | Path | None = None, max_points: int = 2400
) -> pd.DataFrame:
    with connect(database) as connection:
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM activity_streams WHERE activity_id = ?", (activity_id,)
            ).fetchone()[0]
        )
        step = max(1, (count + max_points - 1) // max_points)
        frame = pd.read_sql_query(
            """SELECT * FROM activity_streams
               WHERE activity_id = ? AND (sequence % ? = 0 OR sequence = ?)
               ORDER BY sequence""",
            connection,
            params=(activity_id, step, max(0, count - 1)),
        )
    return frame


def laps(activity_id: int, database: str | Path | None = None) -> pd.DataFrame:
    with connect(database) as connection:
        return pd.read_sql_query(
            "SELECT * FROM activity_laps WHERE activity_id = ? ORDER BY lap_number",
            connection,
            params=(activity_id,),
        )


def power_curve(database: str | Path | None = None, days: int | None = None) -> pd.DataFrame:
    query = """
        SELECT p.duration_seconds, MAX(p.best_power_w) AS best_power_w
        FROM power_curve_results p
        JOIN activities a ON a.id = p.activity_id
    """
    params: list[object] = []
    if days is not None:
        query += " WHERE a.started_at >= datetime('now', ?)"
        params.append(f"-{int(days)} days")
    query += " GROUP BY p.duration_seconds ORDER BY p.duration_seconds"
    with connect(database) as connection:
        return pd.read_sql_query(query, connection, params=params)


def power_history(duration_seconds: int, database: str | Path | None = None) -> pd.DataFrame:
    with connect(database) as connection:
        frame = pd.read_sql_query(
            """SELECT a.id, a.started_at, a.name, p.best_power_w
               FROM power_curve_results p
               JOIN activities a ON a.id = p.activity_id
               WHERE p.duration_seconds = ? ORDER BY a.started_at""",
            connection,
            params=(duration_seconds,),
        )
    if not frame.empty:
        frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True, errors="coerce")
    return frame


def comparable_activities(
    activity_id: int, database: str | Path | None = None, weeks: int = 8
) -> pd.DataFrame:
    target = activity(activity_id, database)
    if not target:
        return pd.DataFrame()
    started = datetime.fromisoformat(str(target["started_at"]))
    duration = float(target.get("moving_seconds") or target.get("elapsed_seconds") or 0)
    distance = float(target.get("distance_m") or 0)
    elevation = float(target.get("ascent_m") or 0)
    with connect(database) as connection:
        candidates = pd.read_sql_query(
            ACTIVITY_SELECT
            + """ WHERE a.sport = ? AND a.id != ?
                  AND a.started_at >= ? AND a.started_at < ?
                  ORDER BY a.started_at DESC""",
            connection,
            params=(
                target["sport"],
                activity_id,
                (started - timedelta(weeks=weeks)).isoformat(),
                started.isoformat(),
            ),
        )
    if candidates.empty:
        return candidates
    candidate_duration = candidates["moving_seconds"].fillna(candidates["elapsed_seconds"]).fillna(0)
    mask = candidate_duration.between(duration * 0.7, duration * 1.3) if duration else candidate_duration >= 0
    if distance > 1000:
        mask &= candidates["distance_m"].fillna(0).between(distance * 0.7, distance * 1.3)
    if elevation > 100:
        mask &= candidates["ascent_m"].fillna(0).between(elevation * 0.5, elevation * 1.5)
    return candidates[mask].copy()


def import_history(database: str | Path | None = None) -> pd.DataFrame:
    init_db(database)
    with connect(database) as connection:
        return pd.read_sql_query(
            """SELECT source_filename, source_format, imported_at, status, activity_count, error
               FROM imports ORDER BY imported_at DESC LIMIT 100""",
            connection,
        )


def data_quality(database: str | Path | None = None) -> dict[str, int]:
    with connect(database) as connection:
        row = connection.execute(
            """SELECT COUNT(*) AS activities,
                      SUM(CASE WHEN avg_hr IS NULL THEN 1 ELSE 0 END) AS missing_hr,
                      SUM(CASE WHEN sport = 'cycling' AND avg_power_w IS NULL THEN 1 ELSE 0 END) AS cycling_missing_power,
                      SUM(CASE WHEN distance_m IS NULL OR distance_m = 0 THEN 1 ELSE 0 END) AS missing_distance
               FROM activities"""
        ).fetchone()
        streams = connection.execute("SELECT COUNT(*) FROM activity_streams").fetchone()[0]
    result = dict(row) if row else {}
    result["trackpoints"] = int(streams)
    return {key: int(value or 0) for key, value in result.items()}


def strength_sets(
    activity_ids: Iterable[int] | None = None, database: str | Path | None = None
) -> pd.DataFrame:
    ids = list(activity_ids or [])
    query = """SELECT s.*, a.started_at AS activity_started_at, a.name AS activity_name
               FROM strength_sets s JOIN activities a ON a.id = s.activity_id"""
    params: list[object] = []
    if ids:
        query += f" WHERE s.activity_id IN ({','.join('?' for _ in ids)})"
        params.extend(ids)
    query += " ORDER BY a.started_at DESC, s.id"
    with connect(database) as connection:
        return pd.read_sql_query(query, connection, params=params)


def decoded_zones(record: dict, key: str) -> list[float]:
    try:
        return [float(value) for value in json.loads(record.get(key) or "[]")]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
