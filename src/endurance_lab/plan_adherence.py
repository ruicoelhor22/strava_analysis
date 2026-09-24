from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from endurance_lab.db import connect, init_db


SESSION_QUERY = """
SELECT p.*, w.week_start, w.week_end, w.phase_name, w.week_type,
       m.activity_id, m.match_score, m.match_method,
       COALESCE(m.match_status, ms.candidate_status) AS match_status,
       m.reason_json,
       a.started_at AS actual_started_at, a.sport AS actual_sport,
       a.name AS actual_name, a.moving_seconds AS actual_moving_seconds,
       a.elapsed_seconds AS actual_elapsed_seconds, a.distance_m AS actual_distance_m,
       a.avg_hr AS actual_avg_hr, a.avg_power_w AS actual_avg_power_w,
       a.metadata_only AS actual_metadata_only,
       d.normalized_power_w AS actual_normalized_power_w,
       d.selected_load AS actual_load
FROM planned_sessions p
LEFT JOIN training_weeks w ON w.id = p.week_id
LEFT JOIN planned_activity_matches m
       ON m.planned_session_id = p.id AND m.is_selected = 1
LEFT JOIN (
    SELECT planned_session_id,
           CASE
               WHEN MAX(CASE WHEN match_status = 'manual' THEN 1 ELSE 0 END) = 1 THEN 'manual'
               WHEN MAX(CASE WHEN match_status = 'matched' THEN 1 ELSE 0 END) = 1 THEN 'matched'
               WHEN MAX(CASE WHEN match_status = 'probable' THEN 1 ELSE 0 END) = 1 THEN 'probable'
               WHEN MAX(CASE WHEN match_status = 'ambiguous' THEN 1 ELSE 0 END) = 1 THEN 'ambiguous'
           END AS candidate_status
    FROM planned_activity_matches
    GROUP BY planned_session_id
) ms ON ms.planned_session_id = p.id
LEFT JOIN activities a ON a.id = m.activity_id
LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
WHERE p.active = 1
"""


def planned_sessions(
    database: str | Path | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    today: date | None = None,
) -> list[dict[str, Any]]:
    init_db(database)
    query = SESSION_QUERY
    params: list[Any] = []
    clauses = []
    if start is not None:
        clauses.append("p.planned_date >= ?")
        params.append(_date(start).isoformat())
    if end is not None:
        clauses.append("p.planned_date <= ?")
        params.append(_date(end).isoformat())
    if clauses:
        query += " AND " + " AND ".join(clauses)
    query += " ORDER BY p.planned_date, p.id"
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(query, params)]
        target_rows = connection.execute(
            "SELECT * FROM planned_session_targets ORDER BY planned_session_id, target_type"
        ).fetchall()
    targets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for target in target_rows:
        targets[int(target["planned_session_id"])].append(dict(target))
    current = today or date.today()
    return [_decorate(row, targets.get(int(row["id"]), []), current) for row in rows]


def weekly_adherence(
    database: str | Path | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    today: date | None = None,
) -> list[dict[str, Any]]:
    sessions = planned_sessions(database, start, end, today)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["week_start"])].append(session)
    result = []
    with connect(database) as connection:
        for week_start, items in sorted(grouped.items()):
            week_end = date.fromisoformat(week_start) + timedelta(days=6)
            actual_rows = connection.execute(
                """SELECT a.id, a.sport, a.started_at,
                          COALESCE(a.moving_seconds, a.elapsed_seconds, 0) AS duration_seconds,
                          COALESCE(a.distance_m, 0) AS distance_m,
                          COALESCE(d.selected_load, 0) AS selected_load,
                          CASE WHEN m.activity_id IS NULL THEN 1 ELSE 0 END AS unmatched
                   FROM activities a
                   LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
                   LEFT JOIN planned_activity_matches m ON m.activity_id = a.id AND m.is_selected = 1
                   WHERE substr(a.started_at, 1, 10) BETWEEN ? AND ?""",
                (week_start, week_end.isoformat()),
            ).fetchall()
            completed = sum(item["completion_state"] in {"completed", "completed_metadata", "partial"} for item in items)
            missed = sum(item["completion_state"] == "missed" for item in items)
            planned_duration = sum(float(item["planned_duration_seconds"] or 0) for item in items)
            actual_duration = sum(float(row["duration_seconds"] or 0) for row in actual_rows)
            planned_distance: dict[str, float] = defaultdict(float)
            actual_distance: dict[str, float] = defaultdict(float)
            for item in items:
                planned_distance[str(item["sport"])] += float(item["planned_distance_m"] or 0)
            for row in actual_rows:
                actual_distance[str(row["sport"])] += float(row["distance_m"] or 0)
            planned_load_values = [float(item["planned_load"]) for item in items if item["planned_load"] is not None]
            key_items = [item for item in items if str(item.get("priority") or "").startswith("A")]
            result.append(
                {
                    "week_start": week_start,
                    "week_end": week_end.isoformat(),
                    "phase": items[0].get("phase_name"),
                    "week_type": items[0].get("week_type"),
                    "planned_sessions": len(items),
                    "completed_sessions": completed,
                    "missed_sessions": missed,
                    "unmatched_activities": sum(int(row["unmatched"]) for row in actual_rows),
                    "planned_duration_seconds": planned_duration,
                    "actual_duration_seconds": actual_duration,
                    "planned_distance_by_sport": dict(planned_distance),
                    "actual_distance_by_sport": dict(actual_distance),
                    "planned_load": sum(planned_load_values) if planned_load_values else None,
                    "actual_load": sum(float(row["selected_load"] or 0) for row in actual_rows),
                    "key_sessions_planned": len(key_items),
                    "key_sessions_completed": sum(
                        item["completion_state"] in {"completed", "completed_metadata", "partial"}
                        for item in key_items
                    ),
                    "strength_sessions_planned": sum(item["sport"] == "strength" for item in items),
                    "strength_sessions_completed": sum(
                        item["sport"] == "strength" and item["completion_state"] in {"completed", "completed_metadata", "partial"}
                        for item in items
                    ),
                    "completion_percentage": completed / len(items) if items else None,
                }
            )
    return result


def plan_status(database: str | Path | None = None, today: date | None = None) -> dict[str, Any]:
    init_db(database)
    current = today or date.today()
    sessions = planned_sessions(database, today=current)
    with connect(database) as connection:
        plan = connection.execute(
            "SELECT * FROM training_plans WHERE active = 1 ORDER BY imported_at DESC LIMIT 1"
        ).fetchone()
    match_counts = defaultdict(int)
    state_counts = defaultdict(int)
    for session in sessions:
        match_counts[session.get("match_status") or "unmatched"] += 1
        state_counts[session["completion_state"]] += 1
    monday = current - timedelta(days=current.weekday())
    current_week = next(
        (week for week in weekly_adherence(database, monday, monday + timedelta(days=6), current)),
        None,
    )
    return {
        "plan": dict(plan) if plan else None,
        "sessions": len(sessions),
        "matches": dict(match_counts),
        "states": dict(state_counts),
        "current_week": current_week,
    }


def reconcile_plan(database: str | Path | None = None, today: date | None = None) -> dict[str, Any]:
    sessions = planned_sessions(database, today=today)
    current = today or date.today()
    return {
        "planned_sessions": len(sessions),
        "matched": sum(item.get("match_status") in {"matched", "manual"} for item in sessions),
        "probable": sum(item.get("match_status") == "probable" for item in sessions),
        "ambiguous": sum(item.get("match_status") == "ambiguous" for item in sessions),
        "unmatched": sum(not item.get("match_status") for item in sessions),
        "unmatched_due": sum(
            not item.get("match_status") and date.fromisoformat(item["planned_date"]) <= current
            and item["completion_state"] not in {"completed_metadata"}
            for item in sessions
        ),
        "upcoming": sum(item["completion_state"] == "upcoming" for item in sessions),
        "metadata_completions": sum(item["completion_state"] == "completed_metadata" for item in sessions),
        "sessions_without_duration": [item["id"] for item in sessions if item["planned_duration_seconds"] is None],
        "low_confidence_targets": [
            {"session_id": item["id"], "target_type": target["target_type"], "raw_text": target["raw_text"]}
            for item in sessions for target in item["targets"] if target["confidence"] == "low"
        ],
    }


def _decorate(row: dict[str, Any], targets: list[dict[str, Any]], today: date) -> dict[str, Any]:
    actual = _json(row.get("workbook_actual_json"))
    planned_date = date.fromisoformat(str(row["planned_date"]))
    source_status = str(row.get("source_status") or "").lower()
    if row.get("activity_id"):
        state = "partial" if _duration_class(row) == "under" and _duration_ratio(row) < 0.7 else "completed"
    elif actual.get("Actual Session") and source_status in {"completed", "modified"}:
        state = "completed_metadata"
    elif source_status == "skipped":
        state = "missed"
    elif planned_date < today and _is_optional_unstructured(row):
        state = "unknown"
    elif planned_date < today:
        state = "missed"
    elif planned_date == today:
        state = "planned"
    else:
        state = "upcoming"
    row["targets"] = targets
    row["workbook_actual"] = actual
    row["completion_state"] = state
    row["duration_adherence"] = _duration_class(row)
    row["distance_adherence"] = _distance_class(row)
    row["intensity_adherence"] = _intensity_class(row, targets)
    row["matching_reason"] = _json(row.get("reason_json"))
    return row


def _actual_duration(row) -> float | None:
    value = row.get("actual_moving_seconds") or row.get("actual_elapsed_seconds")
    if value is not None:
        return float(value)
    minutes = _json(row.get("workbook_actual_json")).get("Actual Duration (min)")
    return float(minutes) * 60 if minutes is not None else None


def _duration_ratio(row) -> float:
    planned = row.get("planned_duration_seconds")
    actual = _actual_duration(row)
    return float(actual) / float(planned) if planned and actual is not None else 1.0


def _duration_class(row) -> str:
    planned = row.get("planned_duration_seconds")
    actual = _actual_duration(row)
    if not planned or actual is None:
        return "unknown"
    ratio = float(actual) / float(planned)
    return "under" if ratio < 0.85 else "over" if ratio > 1.15 else "close"


def _distance_class(row) -> str:
    planned = row.get("planned_distance_m")
    actual = row.get("actual_distance_m")
    if actual is None:
        km = _json(row.get("workbook_actual_json")).get("Distance (km)")
        actual = float(km) * 1000 if km is not None else None
    if not planned or actual is None:
        return "unknown"
    ratio = float(actual) / float(planned)
    return "under" if ratio < 0.85 else "over" if ratio > 1.15 else "close"


def _intensity_class(row, targets) -> str:
    # Whole-activity averages are not comparable with interval-block targets.
    # Keep these sessions unknown until interval-level execution is available.
    session_type = str(row.get("session_type") or "").lower()
    intensity = str(row.get("planned_intensity") or "").lower()
    if any(token in f"{session_type} {intensity}" for token in ("interval", "threshold", "vo2")):
        return "unknown"
    for kind, actual_keys in (
        ("power", ("actual_normalized_power_w", "actual_avg_power_w")),
        ("heart_rate", ("actual_avg_hr",)),
    ):
        target = next((item for item in targets if item["target_type"] == kind), None)
        actual = next((float(row[key]) for key in actual_keys if row.get(key) is not None), None)
        if not target or actual is None or target["minimum_value"] is None or target["maximum_value"] is None:
            continue
        minimum, maximum = float(target["minimum_value"]), float(target["maximum_value"])
        return "easier" if actual < minimum * 0.95 else "harder" if actual > maximum * 1.05 else "close"
    return "unknown"


def _is_optional_unstructured(row) -> bool:
    priority = str(row.get("priority") or "").lower()
    sport = str(row.get("sport") or "").lower()
    title = str(row.get("planned_title") or "").lower()
    return (
        "optional" in priority
        or sport in {"other", "recovery"}
        or any(token in title for token in ("holiday", "unstructured", "rest day"))
    ) and not row.get("planned_duration_seconds")


def _json(value) -> dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
        return loaded if isinstance(loaded, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
