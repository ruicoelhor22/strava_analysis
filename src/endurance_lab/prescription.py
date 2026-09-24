from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from endurance_lab.athlete_state import athlete_state
from endurance_lab.coaching_models import Prescription
from endurance_lab.coaching_rules import RuleDecision, evaluate_rules
from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db


ENGINE_VERSION = 1


def prescribe(
    prescription_date: date,
    database: str | Path | None = None,
    persist: bool = True,
) -> Prescription:
    prescriptions = prescribe_day(prescription_date, database, persist)
    if prescriptions:
        return prescriptions[0]
    return Prescription(
        prescription_date, None, None, None, "NO_PLAN",
        ("No explicit workout is present in the imported coaching plan for this date.",),
        (), "high", ("no_planned_session",), None,
    )


def prescribe_day(
    prescription_date: date,
    database: str | Path | None = None,
    persist: bool = True,
) -> list[Prescription]:
    from endurance_lab.weekly_adjustment import adjust_week

    return [
        item for item in adjust_week(
            prescription_date, database, as_of=prescription_date, persist=persist
        ) if item.date == prescription_date
    ]


def prescribe_week(
    week_date: date,
    database: str | Path | None = None,
    persist: bool = True,
) -> list[Prescription]:
    from endurance_lab.weekly_adjustment import adjust_week

    return adjust_week(week_date, database, as_of=date.today(), persist=persist)


def _sessions_for_date(value: date, database) -> list[dict[str, Any]]:
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT * FROM planned_sessions
               WHERE active = 1 AND planned_date = ?
               ORDER BY CASE WHEN priority LIKE 'A%' THEN 0 WHEN priority LIKE 'B%' THEN 1 ELSE 2 END, id""",
            (value.isoformat(),),
        )]
        for row in rows:
            row["targets"] = [dict(target) for target in connection.execute(
                "SELECT * FROM planned_session_targets WHERE planned_session_id = ? ORDER BY target_type",
                (row["id"],),
            )]
    return rows


def _workout(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": session["planned_date"],
        "sport": session["sport"],
        "title": session["title"],
        "description": session.get("description"),
        "duration_seconds": session.get("planned_duration_seconds"),
        "distance_m": session.get("planned_distance_m"),
        "intensity": session.get("intensity"),
        "interval_structure": session.get("interval_structure"),
        "priority": session.get("priority"),
        "targets": {
            target["target_type"]: target["raw_text"] for target in session.get("targets", [])
        },
    }


def _adapt(original: dict[str, Any], action: str, value: date) -> dict[str, Any]:
    result = dict(original)
    settings = load_athlete_config().get("coaching", {}).get("rules", {})
    duration = result.get("duration_seconds")
    if action == "REDUCE_DURATION" and duration:
        fraction = float(settings.get("duration_reduction_fraction", 0.70))
        minimum = float(settings.get("minimum_prescribed_minutes", 30)) * 60
        quality_minimum = _quality_session_minimum(original)
        result["duration_seconds"] = min(
            float(duration), max(minimum, quality_minimum, round(float(duration) * fraction / 300) * 300)
        )
        result["title"] = f"Shortened: {original['title']}"
        result["description"] = (
            "Prioritize the prescribed main work and remove only optional volume. "
            + str(original.get("description") or "")
        ).strip()
    elif action == "REDUCE_INTENSITY":
        result["title"] = f"Easy alternative: {original['title']}"
        result["intensity"] = "Easy / conversational endurance"
        result["interval_structure"] = None
        result["targets"] = {}
    elif action == "RECOVERY_SESSION":
        result.update({
            "title": "Recovery session", "description": "Very easy movement; stop if it does not feel restorative.",
            "duration_seconds": min(float(duration or 2700), 2700), "distance_m": None,
            "intensity": "Very easy", "interval_structure": None, "targets": {},
        })
    elif action == "REST":
        result.update({
            "title": "Rest", "description": "No structured training prescribed.",
            "duration_seconds": 0, "distance_m": None, "intensity": None,
            "interval_structure": None, "targets": {},
        })
    elif action == "MOVE_SESSION":
        result["date"] = (value + timedelta(days=1)).isoformat()
        result["title"] = f"Move one day: {original['title']}"
    return result


def _quality_session_minimum(workout: dict[str, Any]) -> float:
    text = " ".join(
        str(workout.get(key) or "") for key in ("title", "description", "interval_structure")
    ).lower()
    match = re.search(r"\b(\d{1,2})\s*[x×]\s*(\d{1,3})\s*(?:min|')?", text)
    if not match:
        return 0.0
    repeats, work_minutes = int(match.group(1)), int(match.group(2))
    return float((repeats * work_minutes + max(0, repeats - 1) * 5 + 15) * 60)


def _protect_next_day(session, value: date, decision: RuleDecision, database) -> RuleDecision:
    if decision.action != "MOVE_SESSION":
        return decision
    tomorrow = _sessions_for_date(value + timedelta(days=1), database)
    if not tomorrow:
        return decision
    return RuleDecision(
        "REDUCE_INTENSITY", "high",
        decision.reasons + ("The next day already contains planned training, so the missed work is not stacked there.",),
        decision.evidence + ({"next_day_planned_sessions": len(tomorrow)},),
        decision.rules + ("do_not_stack_moved_session",),
    )


def _persist(item: Prescription, database) -> int:
    now = datetime.now(timezone.utc).isoformat()
    original_json = json.dumps(item.original, ensure_ascii=False, sort_keys=True)
    prescribed_json = json.dumps(item.prescribed, ensure_ascii=False, sort_keys=True)
    evidence_json = json.dumps(item.evidence, ensure_ascii=False, sort_keys=True)
    rules_json = json.dumps(item.rules_triggered, ensure_ascii=False)
    with connect(database) as connection:
        current = connection.execute(
            """SELECT * FROM workout_prescriptions
               WHERE planned_session_id = ? AND status = 'active'""",
            (item.planned_session_id,),
        ).fetchone()
        if current and (
            current["action"] == item.action
            and current["prescribed_json"] == prescribed_json
            and current["evidence_json"] == evidence_json
        ):
            return int(current["id"])
        if current:
            superseded_status = f"superseded:{now}"
            connection.execute(
                "UPDATE workout_prescriptions SET status = ? WHERE id = ?",
                (superseded_status, current["id"]),
            )
        cursor = connection.execute(
            """INSERT INTO workout_prescriptions (
                   planned_session_id, prescribed_date, sport, title, description,
                   duration_seconds, distance_m, intensity, targets_json, reason,
                   status, action, confidence, original_json, prescribed_json,
                   evidence_json, rules_json, as_of_date, engine_version, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.planned_session_id, item.date.isoformat(), item.prescribed["sport"],
                item.prescribed["title"], item.prescribed.get("description"),
                item.prescribed.get("duration_seconds"), item.prescribed.get("distance_m"),
                item.prescribed.get("intensity"), json.dumps(item.prescribed.get("targets", {}), ensure_ascii=False),
                " ".join(item.reasons), item.action, item.confidence, original_json,
                prescribed_json, evidence_json, rules_json, item.date.isoformat(), ENGINE_VERSION, now,
            ),
        )
        new_id = int(cursor.lastrowid)
        if current:
            connection.execute(
                "UPDATE workout_prescriptions SET superseded_by = ? WHERE id = ?",
                (new_id, current["id"]),
            )
        connection.commit()
    return new_id
