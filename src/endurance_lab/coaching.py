from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db
from endurance_lab.plan_adherence import planned_sessions


def coaching_flags(
    database: str | Path | None = None,
    today: date | None = None,
) -> list[dict[str, Any]]:
    init_db(database)
    current = today or date.today()
    start = current - timedelta(days=28)
    sessions = planned_sessions(database, start, current, current)
    flags: list[dict[str, Any]] = []

    for session in sessions:
        if session["completion_state"] == "missed" and str(session.get("priority") or "").startswith("A"):
            flags.append(_flag(
                "missed_key_session", "attention",
                f"Key {session['sport']} session on {session['planned_date']} has no completed match.",
                session_id=session["id"],
            ))

    current_week_start = current - timedelta(days=current.weekday())
    completed_due = [
        item for item in sessions
        if item["planned_date"] <= current.isoformat() and item["completion_state"] != "unknown"
    ]
    completed = [item for item in completed_due if item["completion_state"] in {"completed", "completed_metadata", "partial"}]
    if len(completed_due) >= 4 and len(completed) / len(completed_due) < 0.6:
        flags.append(_flag(
            "low_recent_training_consistency", "watch",
            f"{len(completed)} of {len(completed_due)} planned sessions were completed in the last 28 days.",
            value=len(completed) / len(completed_due),
        ))

    hard_limit = float(load_athlete_config().get("load_model", {}).get("hard_session_load", 100))
    with connect(database) as connection:
        actuals = [dict(row) for row in connection.execute(
            """SELECT a.id, substr(a.started_at, 1, 10) AS day, a.sport, a.name,
                      COALESCE(a.moving_seconds, a.elapsed_seconds, 0) AS duration_seconds,
                      COALESCE(d.selected_load, 0) AS selected_load,
                      CASE WHEN m.activity_id IS NULL THEN 1 ELSE 0 END AS unmatched
               FROM activities a
               LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
               LEFT JOIN planned_activity_matches m ON m.activity_id = a.id AND m.is_selected = 1
               WHERE substr(a.started_at, 1, 10) BETWEEN ? AND ? ORDER BY a.started_at""",
            (start.isoformat(), current.isoformat()),
        )]
        loads = [dict(row) for row in connection.execute(
            "SELECT day, total_load FROM daily_training_load WHERE day BETWEEN ? AND ? ORDER BY day",
            ((current - timedelta(days=14)).isoformat(), current.isoformat()),
        )]

    planned_to_date = sum(
        float(item["planned_duration_seconds"] or 0)
        for item in sessions
        if current_week_start.isoformat() <= item["planned_date"] <= current.isoformat()
        and item["completion_state"] != "unknown"
    )
    actual_to_date = sum(
        float(item["duration_seconds"] or 0)
        for item in actuals
        if item["day"] >= current_week_start.isoformat()
    )
    if planned_to_date > 0:
        deviation = (actual_to_date - planned_to_date) / planned_to_date
        if abs(deviation) >= 0.25:
            code = "high_actual_vs_planned_volume" if deviation > 0 else "large_volume_deviation"
            flags.append(_flag(
                code, "watch",
                f"Actual duration so far this week is {abs(deviation):.0%} "
                f"{'above' if deviation > 0 else 'below'} the plan through today.",
                value=deviation,
            ))
    hard_days = sorted({item["day"] for item in actuals if float(item["selected_load"] or 0) >= hard_limit})
    for previous, following in zip(hard_days, hard_days[1:]):
        if (date.fromisoformat(following) - date.fromisoformat(previous)).days <= 1:
            flags.append(_flag(
                "multiple_hard_days", "watch",
                f"Hard-load activities occurred on consecutive days ({previous} and {following}).",
            ))
            break

    for item in actuals:
        if item["unmatched"] and float(item["selected_load"] or 0) >= hard_limit:
            flags.append(_flag(
                "extra_hard_session", "watch",
                f"Unmatched {item['sport']} activity on {item['day']} carried estimated load {item['selected_load']:.0f}.",
                activity_id=item["id"],
            ))
        elif item["unmatched"]:
            flags.append(_flag(
                "unmatched_activity", "info",
                f"{item['sport'].title()} activity on {item['day']} is not linked to a planned session.",
                activity_id=item["id"],
            ))

    if loads:
        recent = sum(float(item["total_load"]) for item in loads if item["day"] > (current - timedelta(days=7)).isoformat())
        prior = sum(float(item["total_load"]) for item in loads if item["day"] <= (current - timedelta(days=7)).isoformat())
        if prior >= hard_limit and recent > prior * 1.3:
            flags.append(_flag(
                "rapid_7_day_load_increase", "watch",
                f"Recent 7-day estimated load is {(recent / prior - 1):.0%} above the preceding 7 days.",
                value=recent / prior - 1,
            ))

    key_endurance = [
        item for item in sessions
        if str(item.get("priority") or "").startswith("A") and item["sport"] in {"cycling", "running", "swimming"}
    ]
    strength_days = {item["day"] for item in actuals if item["sport"] == "strength"}
    for session in key_endurance:
        previous = (date.fromisoformat(session["planned_date"]) - timedelta(days=1)).isoformat()
        if previous in strength_days:
            flags.append(_flag(
                "strength_before_key_endurance_session", "info",
                f"Strength training occurred the day before key {session['sport']} work on {session['planned_date']}.",
                session_id=session["id"],
            ))
    return _deduplicate(flags)


def _flag(code: str, severity: str, message: str, **details) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "details": details}


def _deduplicate(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for flag in flags:
        key = (flag["code"], flag["message"])
        if key not in seen:
            seen.add(key)
            result.append(flag)
    return result
