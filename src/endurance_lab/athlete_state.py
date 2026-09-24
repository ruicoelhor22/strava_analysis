from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from endurance_lab.coaching_models import AthleteState, ContextDimension
from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db
from endurance_lab.performance_trends import performance_trends
from endurance_lab.session_classifier import classify_record
from endurance_lab.subjective import latest_checkin


def athlete_state(
    as_of: date,
    database: str | Path | None = None,
) -> AthleteState:
    """Build training context from facts dated strictly before ``as_of``.

    A check-in dated ``as_of`` is included because it can be supplied before that
    day's prescription. Activities and completed-workout evidence on ``as_of`` are
    intentionally excluded.
    """
    init_db(database)
    config = load_athlete_config()
    start = as_of - timedelta(days=42)
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(
            _STATE_ACTIVITY_QUERY,
            (start.isoformat(), as_of.isoformat()),
        )]
        adherence = _adherence(connection, as_of)

    classified = [
        (row, classify_record(row, config, int(row.get("lower_body_sets") or 0)))
        for row in rows
    ]
    recent_training = _recent_training(rows, as_of)
    load = _load_context(rows, as_of)
    recovery = _recovery_context(classified, as_of, config)
    trends = performance_trends(as_of, database)
    performance = {sport: signal.to_dict() for sport, signal in trends.items()}
    quality = _data_quality(rows)
    subjective = latest_checkin(as_of, database)
    dimensions = _dimensions(load, recovery, trends, adherence, quality, config)
    return AthleteState(
        as_of, recent_training, load, recovery, performance, adherence,
        quality, subjective, dimensions,
    )


def _recent_training(rows: list[dict[str, Any]], as_of: date) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for days in (7, 28, 42):
        cutoff = as_of - timedelta(days=days)
        selected = [row for row in rows if date.fromisoformat(row["day"]) >= cutoff]
        sport_seconds: dict[str, float] = defaultdict(float)
        for row in selected:
            sport_seconds[str(row["sport"])] += float(row["duration_seconds"] or 0)
        windows[f"{days}d"] = {
            "duration_seconds": sum(float(row["duration_seconds"] or 0) for row in selected),
            "sessions": len(selected),
            "sport_duration_seconds": dict(sport_seconds),
        }
    return windows


def _load_context(rows: list[dict[str, Any]], as_of: date) -> dict[str, Any]:
    load_7d = _window_sum(rows, as_of, 7, "selected_load")
    load_28d = _window_sum(rows, as_of, 28, "selected_load")
    load_42d = _window_sum(rows, as_of, 42, "selected_load")
    reference_start = as_of - timedelta(days=28)
    reference_end = as_of - timedelta(days=7)
    reference_total = sum(
        float(row.get("selected_load") or 0)
        for row in rows
        if reference_start <= date.fromisoformat(row["day"]) < reference_end
    )
    weekly_reference = reference_total / 3 if reference_total > 0 else None
    ratio = load_7d / weekly_reference if weekly_reference else None
    return {
        "load_7d": load_7d,
        "load_28d": load_28d,
        "load_42d": load_42d,
        "weekly_reference_load": weekly_reference,
        "recent_to_reference_ratio": ratio,
        "high_intensity_sessions_7d": sum(
            1 for row in rows
            if date.fromisoformat(row["day"]) >= as_of - timedelta(days=7)
            and float(row.get("intensity_factor") or 0) >= 0.9
        ),
    }


def _recovery_context(classified, as_of: date, config: dict[str, Any]) -> dict[str, Any]:
    last_hard: dict[str, datetime] = {}
    last_long: date | None = None
    last_leg_strength: datetime | None = None
    training_days: set[date] = set()
    long_thresholds = config.get("load_model", {}).get("long_session_minutes", {})
    for row, result in classified:
        day = date.fromisoformat(row["day"])
        started = datetime.fromisoformat(str(row["started_at"]))
        training_days.add(day)
        if result.stress_level == "high":
            key = result.classification if result.classification == "STRENGTH" else str(row["sport"])
            previous = last_hard.get(key)
            last_hard[key] = max(previous, started) if previous else started
        long_minutes = float(long_thresholds.get(str(row["sport"]), 10_000))
        if float(row.get("duration_seconds") or 0) / 60 >= long_minutes:
            last_long = max(last_long, day) if last_long else day
        if result.classification == "STRENGTH" and result.muscular_load == "high":
            last_leg_strength = max(last_leg_strength, started) if last_leg_strength else started
    consecutive = 0
    cursor = as_of - timedelta(days=1)
    while cursor in training_days:
        consecutive += 1
        cursor -= timedelta(days=1)
    rest_days = sum(
        as_of - timedelta(days=offset) not in training_days for offset in range(1, 8)
    )
    timezone_name = config.get("athlete", {}).get("timezone", "UTC")
    as_of_time = datetime.combine(as_of, time.min, ZoneInfo(timezone_name)).astimezone(timezone.utc)
    return {
        "days_since_hard_ride": _days_since_datetime(last_hard.get("cycling"), as_of),
        "days_since_hard_run": _days_since_datetime(last_hard.get("running"), as_of),
        "days_since_hard_swim": _days_since_datetime(last_hard.get("swimming"), as_of),
        "hours_since_hard_ride": _hours_since(last_hard.get("cycling"), as_of_time),
        "hours_since_hard_run": _hours_since(last_hard.get("running"), as_of_time),
        "hours_since_hard_swim": _hours_since(last_hard.get("swimming"), as_of_time),
        "days_since_long_session": _days_since(last_long, as_of),
        "days_since_leg_strength": _days_since_datetime(last_leg_strength, as_of),
        "hours_since_leg_strength": _hours_since(last_leg_strength, as_of_time),
        "consecutive_training_days": consecutive,
        "rest_days_7d": rest_days,
        "last_activity_date": max(training_days).isoformat() if training_days else None,
        "configured_consecutive_days_watch": int(
            config.get("coaching", {}).get("rules", {}).get("consecutive_training_days_watch", 6)
        ),
    }


def _adherence(connection, as_of: date) -> dict[str, Any]:
    start = as_of - timedelta(days=28)
    rows = connection.execute(
        """SELECT p.id, p.planned_date, p.priority, p.planned_duration_seconds,
                  p.source_status, a.id AS activity_id,
                  COALESCE(a.moving_seconds, a.elapsed_seconds) AS actual_duration
           FROM planned_sessions p
           LEFT JOIN planned_activity_matches m
                  ON m.planned_session_id = p.id AND m.is_selected = 1
           LEFT JOIN activities a ON a.id = m.activity_id AND a.started_at < ?
           WHERE p.active = 1 AND p.planned_date >= ? AND p.planned_date < ?""",
        (as_of.isoformat(), start.isoformat(), as_of.isoformat()),
    ).fetchall()
    planned = len(rows)
    completed = sum(bool(row["activity_id"]) or str(row["source_status"] or "").lower() in {"completed", "modified"} for row in rows)
    missed_key = sum(
        not row["activity_id"] and str(row["priority"] or "").startswith("A")
        and str(row["source_status"] or "").lower() not in {"completed", "modified"}
        for row in rows
    )
    seven_start = as_of - timedelta(days=7)
    recent = [row for row in rows if date.fromisoformat(row["planned_date"]) >= seven_start]
    planned_duration = sum(float(row["planned_duration_seconds"] or 0) for row in recent)
    actual_duration = float(connection.execute(
        """SELECT COALESCE(SUM(COALESCE(moving_seconds, elapsed_seconds, 0)), 0)
           FROM activities WHERE started_at >= ? AND started_at < ?""",
        (seven_start.isoformat(), as_of.isoformat()),
    ).fetchone()[0])
    return {
        "planned_sessions_28d": planned,
        "completed_sessions_28d": completed,
        "completion_ratio_28d": completed / planned if planned else None,
        "missed_key_sessions_28d": missed_key,
        "planned_duration_7d": planned_duration,
        "planned_sessions_7d": len(recent),
        "matched_actual_duration_7d": actual_duration,
        "actual_to_planned_duration_ratio_7d": actual_duration / planned_duration if planned_duration else None,
    }


def _data_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    hr = sum(row.get("avg_hr") is not None for row in rows)
    power = sum(row.get("avg_power_w") is not None or row.get("normalized_power_w") is not None for row in rows)
    streams = sum(int(row.get("stream_points") or 0) > 0 for row in rows)
    confidence = "high" if count >= 8 and streams / count >= 0.7 else "medium" if count >= 4 else "low"
    return {
        "activities_42d": count,
        "heart_rate_availability": hr / count if count else None,
        "power_availability": power / count if count else None,
        "stream_availability": streams / count if count else None,
        "confidence": confidence,
    }


def _dimensions(load, recovery, trends, adherence, quality, config) -> dict[str, ContextDimension]:
    ratio = load.get("recent_to_reference_ratio")
    load_watch = float(
        config.get("coaching", {}).get("rules", {}).get("recent_load_ratio_watch", 1.35)
    )
    load_concern = load_watch + 0.25
    if ratio is None:
        load_dimension = ContextDimension("unknown", ("insufficient reference load",))
    elif ratio >= load_concern:
        load_dimension = ContextDimension("concern", (f"7-day load is {ratio:.2f} times the 28-day weekly reference",))
    elif ratio >= load_watch:
        load_dimension = ContextDimension("watch", (f"7-day load is {ratio:.2f} times the 28-day weekly reference",))
    else:
        load_dimension = ContextDimension("normal", (f"7-day/reference load ratio is {ratio:.2f}",))

    consecutive = int(recovery["consecutive_training_days"])
    threshold = int(recovery["configured_consecutive_days_watch"])
    hard_recent = min(
        value for value in (
            recovery["days_since_hard_ride"], recovery["days_since_hard_run"], recovery["days_since_hard_swim"]
        ) if value is not None
    ) if any(recovery[key] is not None for key in ("days_since_hard_ride", "days_since_hard_run", "days_since_hard_swim")) else None
    if consecutive >= threshold + 1:
        recovery_dimension = ContextDimension("concern", (f"{consecutive} consecutive training days",))
    elif consecutive >= threshold or hard_recent == 1:
        reasons = [f"{consecutive} consecutive training days"] if consecutive >= threshold else []
        if hard_recent == 1:
            reasons.append("a high-stress session occurred yesterday")
        recovery_dimension = ContextDimension("watch", tuple(reasons))
    else:
        recovery_dimension = ContextDimension("normal", ("recent hard-session spacing is not exceptional",))

    trend_values = list(trends.values())
    if any(item.state == "DECLINING" and item.confidence != "low" for item in trend_values):
        trend_dimension = ContextDimension("watch", tuple(
            f"{item.sport} trend is declining" for item in trend_values if item.state == "DECLINING"
        ))
    elif any(item.state == "IMPROVING" and item.confidence != "low" for item in trend_values):
        trend_dimension = ContextDimension("positive", tuple(
            f"{item.sport} trend is improving" for item in trend_values if item.state == "IMPROVING"
        ))
    elif all(item.state == "UNCERTAIN" for item in trend_values):
        trend_dimension = ContextDimension("unknown", ("comparable performance data is insufficient",))
    else:
        trend_dimension = ContextDimension("normal", ("available sport trends are stable or uncertain",))

    completion = adherence.get("completion_ratio_28d")
    if completion is None:
        adherence_dimension = ContextDimension("unknown", ("no due planned sessions in the comparison window",))
    elif completion < 0.6:
        adherence_dimension = ContextDimension("watch", (f"28-day plan completion is {completion:.0%}",))
    elif completion >= 0.85:
        adherence_dimension = ContextDimension("positive", (f"28-day plan completion is {completion:.0%}",))
    else:
        adherence_dimension = ContextDimension("normal", (f"28-day plan completion is {completion:.0%}",))
    quality_dimension = ContextDimension(
        "normal" if quality["confidence"] in {"high", "medium"} else "watch",
        (f"training-context data confidence is {quality['confidence']}",),
    )
    return {
        "LOAD_CONTEXT": load_dimension,
        "RECOVERY_SPACING": recovery_dimension,
        "PERFORMANCE_TREND": trend_dimension,
        "PLAN_ADHERENCE": adherence_dimension,
        "DATA_CONFIDENCE": quality_dimension,
    }


def _window_sum(rows, as_of: date, days: int, field: str) -> float:
    cutoff = as_of - timedelta(days=days)
    return sum(float(row.get(field) or 0) for row in rows if date.fromisoformat(row["day"]) >= cutoff)


def _days_since(previous: date | None, as_of: date) -> int | None:
    return (as_of - previous).days if previous else None


def _days_since_datetime(previous: datetime | None, as_of: date) -> int | None:
    return (as_of - previous.date()).days if previous else None


def _hours_since(previous: datetime | None, as_of: datetime) -> float | None:
    if previous is None:
        return None
    aware = previous if previous.tzinfo else previous.replace(tzinfo=timezone.utc)
    return max(0.0, (as_of - aware.astimezone(timezone.utc)).total_seconds() / 3600)


_STATE_ACTIVITY_QUERY = """
SELECT a.id, substr(a.started_at, 1, 10) AS day, a.started_at, a.sport, a.name,
       a.avg_hr, a.avg_power_w, a.distance_m,
       COALESCE(a.moving_seconds, a.elapsed_seconds, 0) AS duration_seconds,
       d.normalized_power_w, d.intensity_factor, d.selected_load,
       d.efficiency_factor, d.aerobic_decoupling_pct, d.late_fade_pct,
       d.pace_seconds_per_km, d.hr_zones_json, d.power_zones_json,
       p.title AS planned_title, p.session_type AS planned_type,
       p.intensity AS planned_intensity, p.description AS planned_description,
       (SELECT COUNT(*) FROM activity_streams s WHERE s.activity_id = a.id) AS stream_points,
       (SELECT COUNT(*) FROM strength_sets ss WHERE ss.activity_id = a.id AND (
           lower(ss.exercise_name) LIKE '%squat%' OR lower(ss.exercise_name) LIKE '%deadlift%'
           OR lower(ss.exercise_name) LIKE '%leg%' OR lower(ss.exercise_name) LIKE '%lunge%'
           OR lower(ss.exercise_name) LIKE '%calf%' OR lower(ss.exercise_name) LIKE '%glute%'
           OR lower(ss.exercise_name) LIKE '%hip%' OR lower(ss.exercise_name) LIKE '%hamstring%'
           OR lower(ss.exercise_name) LIKE '%quad%'
       )) AS lower_body_sets
FROM activities a
LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
LEFT JOIN planned_activity_matches m ON m.activity_id = a.id AND m.is_selected = 1
LEFT JOIN planned_sessions p ON p.id = m.planned_session_id
WHERE a.started_at >= ? AND a.started_at < ?
ORDER BY a.started_at
"""
