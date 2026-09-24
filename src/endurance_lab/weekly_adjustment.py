from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from endurance_lab.coaching_models import OptionalGate, Prescription, SessionCost
from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db
from endurance_lab.session_cost import session_cost
from endurance_lab.subjective import latest_checkin


ENGINE_VERSION = 2


def adjust_week(
    week_date: date,
    database: str | Path | None = None,
    *,
    as_of: date | None = None,
    persist: bool = True,
) -> list[Prescription]:
    """Reconcile actual cost against the remaining plan without future-data leakage."""
    init_db(database)
    config = load_athlete_config()
    as_of = as_of or date.today()
    monday = week_date - timedelta(days=week_date.weekday())
    sunday = monday + timedelta(days=6)
    sessions = _planned_sessions(monday, sunday, database)
    actual = _actual_context(monday - timedelta(days=7), min(as_of, sunday + timedelta(days=1)), database, persist)
    subjective = latest_checkin(as_of, database)
    results: list[Prescription] = []
    for session in sessions:
        result = _decide(session, sessions, actual, as_of, subjective, config)
        if persist:
            result = replace(result, prescription_id=_persist(result, database))
        results.append(result)
    return results


def reconcile_coaching(
    database: str | Path | None = None,
    *,
    as_of: date | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    as_of = as_of or date.today()
    prescriptions = adjust_week(as_of, database, as_of=as_of, persist=persist)
    return {
        "as_of": as_of.isoformat(),
        "week_start": (as_of - timedelta(days=as_of.weekday())).isoformat(),
        "prescriptions": [item.to_dict() for item in prescriptions],
    }


def explain_date(value: date, database: str | Path | None = None) -> list[dict[str, Any]]:
    return [
        item.to_dict() for item in adjust_week(value, database, as_of=value, persist=False)
        if item.date == value
    ]


def _decide(session, sessions, actual, as_of, subjective, config) -> Prescription:
    planned_date = date.fromisoformat(session["planned_date"])
    original = _workout(session)
    priority, priority_source = _priority(session, config)
    preceding = [
        item for item in actual
        if item["started_at"].date() < planned_date
        and item["started_at"] >= _day_start(planned_date, config) - timedelta(days=7)
    ]
    same_day_actual = [item for item in actual if item["started_at"].date() == planned_date]
    future_keys = [
        item for item in sessions
        if item["planned_date"] > session["planned_date"] and _priority(item, config)[0] == "A"
    ]
    protected = tuple({
        "planned_session_id": int(item["id"]), "date": item["planned_date"],
        "title": item["title"], "sport": item["sport"], "priority": "A",
    } for item in future_keys)
    stressors = [item for item in preceding if _is_meaningful_stressor(item["cost"])]
    latest = max(stressors, key=lambda item: item["started_at"], default=None)
    next_key = future_keys[0] if future_keys else (session if priority == "A" else None)
    runway = _runway(latest, next_key, config)
    interference = _interference(session, preceding, config)
    gate = None
    action = "KEEP"
    rules = ["plan_is_anchor", "minimum_necessary_adjustment"]
    reasons = ["The original plan remains compatible with the measured training cost."]

    completed = bool(session.get("activity_id")) and planned_date < as_of
    subjective_threshold = int(config.get("coaching", {}).get("rules", {}).get("subjective_high_threshold", 4))
    fatigue = int((subjective or {}).get("fatigue") or 0)
    soreness = int((subjective or {}).get("leg_soreness") or 0)
    if completed:
        reasons = ["The session is already completed; actual cost is used to adjust only the remaining week."]
        rules = ["actual_training_over_calendar_compliance"]
    elif planned_date < as_of:
        action = "SKIP"
        reasons = ["The planned session has passed and is not automatically rescheduled as training debt."]
        rules = ["missed_training_is_not_debt"]
    elif fatigue >= 5 and soreness >= 5:
        action = "REST"
        reasons = ["Maximum reported fatigue and leg soreness make recovery the minimum safe intervention."]
        rules = ["subjective_recovery_gate", "rest_when_systemically_compromised"]
    elif max(fatigue, soreness) >= subjective_threshold and not _is_easy_session(session):
        action = "REDUCE_INTENSITY"
        reasons = ["Reported fatigue or soreness is high enough to remove quality intensity while preserving routine."]
        rules = ["subjective_recovery_gate", "minimum_necessary_adjustment"]
    elif priority == "C" and _follows_key_session(session, sessions, config):
        action = "CONDITIONAL"
        gate = _optional_gate(subjective, "Post-key-session recovery must justify optional training.")
        reasons = [
            "This optional session follows an A-priority key session.",
            "It proceeds only when legs and general recovery are fresh or acceptable; otherwise rest.",
        ]
        rules = ["protect_key_session_recovery", "optional_gate"]
    elif interference:
        direct_lower = any(flag["type"] == "lower_body_strength" for flag in interference)
        same_modality_high = any(flag["type"] == "same_modality_high_cost" for flag in interference)
        if str(session["sport"]) == "strength" and direct_lower and priority != "A":
            action = "REPLACE_WITH_EASY"
            gate = _optional_gate(subjective, "Recent lower-body strength makes another gym session redundant.")
            reasons = [
                "Recent strength produced meaningful lower-body muscular load.",
                "A second support-strength session inside the interference window adds limited value.",
            ]
            if protected:
                reasons.append(f"The smaller change protects {protected[0]['title']} ({protected[0]['date']}).")
            rules = ["lower_body_strength_interference", "modify_support_before_key", "minimum_necessary_adjustment"]
        elif _is_easy_session(session):
            if direct_lower and planned_date == as_of:
                action = "CONDITIONAL"
                gate = _optional_gate(subjective, "Easy training is useful only if it feels restorative.")
                reasons = [
                    "Recent lower-body strength creates local muscular cost, while this session is low priority physiologically.",
                    "Keep it genuinely easy only with acceptable legs; otherwise take rest.",
                ]
                rules = ["lower_body_strength_interference", "preserve_easy_day", "optional_gate"]
        elif direct_lower and not _is_easy_session(session):
            action = "REDUCE_INTENSITY"
            reasons = [
                "Recent lower-body strength overlaps with the planned quality session.",
                "Keep the session slot but remove demanding targets to avoid compounding local muscular cost.",
            ]
            rules = ["leg_strength_interference", "minimum_necessary_adjustment"]
        elif same_modality_high and not _session_on_date(sessions, planned_date + timedelta(days=1)):
            action = "MOVE"
            reasons = [
                "A recent high-cost session in the same sport leaves inadequate spacing.",
                "Moving the session by one day preserves its purpose with more recovery runway.",
            ]
            rules = ["high_stress_spacing", "protect_key_session"]
        elif priority == "A":
            if runway is not None and runway < _runway_threshold(config, "high_overlap", 24):
                action = "REDUCE_DURATION"
                reasons = [
                    "A-priority work is protected, but overlapping high cost leaves insufficient recovery runway.",
                    "Reduce optional volume while retaining the session's main purpose.",
                ]
                rules = ["protect_key_session", "insufficient_recovery_runway"]
        else:
            action = "REPLACE_WITH_EASY"
            reasons = [
                "Measured preceding cost overlaps with this support session.",
                "Replacing the lower-priority stress is the smallest change that restores spacing.",
            ]
            rules = ["modality_interference", "modify_support_before_key"]

    if action == "KEEP" and not interference:
        recent_cost_count = sum(
            item["cost"].systemic_cost in {"moderate", "high"}
            for item in preceding if (planned_date - item["started_at"].date()).days <= 7
        )
        if recent_cost_count >= 5 and not _is_easy_session(session):
            action = "REDUCE_DURATION"
            reasons = [
                "The recent sequence contains repeated moderate-or-higher systemic cost.",
                "Trim optional volume while preserving the intended intensity stimulus.",
            ]
            rules = ["accumulated_training_cost", "minimum_necessary_adjustment"]

    if action == "KEEP" and priority == "A":
        rules.append("protect_key_session")
        reasons = ["This A-priority session remains a protected driver of the week's intended adaptation."]
        if runway is not None:
            reasons.append(f"Measured recovery runway is approximately {runway:.0f} hours.")
    if action == "KEEP" and _is_easy_session(session):
        rules.append("preserve_easy_day")
        reasons.append("No missed volume is added; the prescribed easy intensity remains easy.")

    prescribed = _adapt(original, action, planned_date, gate, config)
    trace = {
        "planned_session": {**original, "priority": priority, "priority_source": priority_source},
        "actual_context": [_actual_dict(item) for item in preceding + same_day_actual],
        "cost_signals": [item["cost"].to_dict() for item in preceding],
        "interference_flags": interference,
        "protected_sessions": list(protected),
        "rules_triggered": rules,
        "candidate_actions": _candidate_actions(action),
        "selected_action": action,
        "reasons": reasons,
        "confidence": _confidence(preceding, subjective),
        "recovery_runway_h": runway,
        "optional_gate": gate.to_dict() if gate else None,
    }
    evidence = ({
        "session_costs": [item["cost"].to_dict() for item in preceding],
        "interference_flags": interference,
        "recovery_runway_h": runway,
        "priority": priority,
        "priority_source": priority_source,
    },)
    return Prescription(
        planned_date, int(session["id"]), original, prescribed, action,
        tuple(reasons), evidence, trace["confidence"], tuple(rules), None,
        gate, trace, runway, protected,
    )


def _actual_context(start: date, end: date, database, persist: bool) -> list[dict[str, Any]]:
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT id, source_activity_id, name, sport, started_at,
                      COALESCE(moving_seconds, elapsed_seconds, 0) AS duration_seconds
               FROM activities WHERE started_at >= ? AND started_at < ? ORDER BY started_at""",
            (start.isoformat(), end.isoformat()),
        )]
    return [
        {**row, "started_at": datetime.fromisoformat(row["started_at"]),
         "cost": session_cost(int(row["id"]), database, persist=persist)}
        for row in rows
    ]


def _planned_sessions(start: date, end: date, database) -> list[dict[str, Any]]:
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT p.*, a.id AS activity_id, a.name AS actual_name
               FROM planned_sessions p
               LEFT JOIN planned_activity_matches m ON m.planned_session_id=p.id AND m.is_selected=1
               LEFT JOIN activities a ON a.id=m.activity_id
               WHERE p.active=1 AND p.planned_date>=? AND p.planned_date<=?
               ORDER BY p.planned_date, p.id""",
            (start.isoformat(), end.isoformat()),
        )]
        for row in rows:
            row["targets"] = [dict(item) for item in connection.execute(
                "SELECT * FROM planned_session_targets WHERE planned_session_id=? ORDER BY target_type",
                (row["id"],),
            )]
    return rows


def _interference(session, preceding, config) -> list[dict[str, Any]]:
    planned_start = _day_start(date.fromisoformat(session["planned_date"]), config) + timedelta(hours=9)
    sport = str(session["sport"])
    flags = []
    for item in preceding:
        hours = (planned_start - item["started_at"]).total_seconds() / 3600
        if hours < 0 or hours > 96:
            continue
        cost = item["cost"]
        if cost.sport_specific_cost == "strength" and cost.muscle_load in {"lower", "full"} and cost.muscular_cost in {"moderate", "high"} and sport in {"cycling", "running", "strength"}:
            threshold = _runway_threshold(config, "meaningful_overlap", 48)
            if hours <= threshold:
                flags.append({
                    "type": "lower_body_strength", "activity_id": cost.activity_id,
                    "hours_before": round(hours, 1), "severity": "high" if hours < 24 else "meaningful",
                    "evidence": f"{cost.muscular_cost} {cost.muscle_load} muscular cost",
                })
        if cost.systemic_cost == "high" and cost.sport_specific_cost == sport:
            threshold = _runway_threshold(config, "meaningful_overlap", 48)
            if hours <= threshold:
                flags.append({
                    "type": "same_modality_high_cost", "activity_id": cost.activity_id,
                    "hours_before": round(hours, 1), "severity": "high" if hours < 24 else "meaningful",
                    "evidence": f"high {sport}-specific cost",
                })
    return flags


def _priority(session, config=None) -> tuple[str, str]:
    explicit = str(session.get("priority") or "").strip().upper()
    if explicit[:1] in {"A", "B", "C"}:
        return explicit[0], str(session.get("priority_source") or "workbook")
    defaults = (config or {}).get("coaching", {}).get("priority_defaults", {})
    a_terms = tuple(defaults.get("a", ("threshold", "vo2", "race", "long")))
    c_terms = tuple(defaults.get("c", ("recovery", "optional", "easy spin")))
    text = " ".join(str(session.get(key) or "") for key in ("title", "session_type", "intensity")).lower()
    if any(token in text for token in a_terms):
        return "A", "inferred"
    if any(token in text for token in c_terms):
        return "C", "inferred"
    return "B", "inferred"


def _workout(session) -> dict[str, Any]:
    return {
        "date": session["planned_date"], "sport": session["sport"], "title": session["title"],
        "description": session.get("description"), "duration_seconds": session.get("planned_duration_seconds"),
        "distance_m": session.get("planned_distance_m"), "intensity": session.get("intensity"),
        "interval_structure": session.get("interval_structure"), "priority": session.get("priority"),
        "targets": {target["target_type"]: target["raw_text"] for target in session.get("targets", [])},
    }


def _adapt(original, action, value, gate, config) -> dict[str, Any] | None:
    if action == "SKIP":
        return None
    result = dict(original)
    if action == "REST":
        result.update({
            "sport": "other", "title": "Rest", "description": "Recovery selected by the coaching gate.",
            "duration_seconds": 0, "distance_m": None, "intensity": "Rest",
            "interval_structure": None, "targets": {},
        })
    if action == "REDUCE_DURATION" and result.get("duration_seconds"):
        rule_config = config.get("coaching", {}).get("rules", {})
        fraction = float(rule_config.get("duration_reduction_fraction", 0.70))
        minimum = float(rule_config.get("minimum_prescribed_minutes", 30)) * 60
        result["duration_seconds"] = max(minimum, round(float(result["duration_seconds"]) * fraction / 300) * 300)
        result["title"] = f"Shortened: {result['title']}"
    elif action == "REDUCE_INTENSITY":
        result["title"] = f"Easy version: {result['title']}"
        result["description"] = "Keep this session conversational and remove quality intervals or hard targets."
        result["intensity"] = "Easy / recovery"
        result["interval_structure"] = None
        result["targets"] = {}
    elif action == "MOVE":
        result["date"] = (value + timedelta(days=1)).isoformat()
        result["title"] = f"Moved: {result['title']}"
    elif action == "REPLACE_WITH_EASY":
        result.update({
            "sport": "running" if original.get("sport") == "strength" else original.get("sport"),
            "title": "Easy run / rest gate" if original.get("sport") == "strength" else "Easy recovery alternative",
            "description": "30–45 min genuinely easy and conversational if legs are acceptable; otherwise rest.",
            "duration_seconds": 2400, "distance_m": None, "intensity": "Recovery / easy aerobic",
            "interval_structure": None, "targets": {},
        })
    elif action == "CONDITIONAL":
        result["title"] = f"Conditional: {result['title']}"
        result["description"] = (str(result.get("description") or "") + " Proceed only when the recovery gate is met; otherwise rest.").strip()
    return result


def _optional_gate(subjective, reason: str) -> OptionalGate:
    if not subjective:
        state = "unknown"
        signals = ("No subjective recovery check-in is available.", reason)
    else:
        notes = str(subjective.get("notes") or "").lower()
        if any(token in notes for token in ("ill", "sick", "pain", "injury", "fever")):
            state = "symptomatic"
        elif max(int(subjective.get("fatigue") or 0), int(subjective.get("leg_soreness") or 0)) >= 4:
            state = "heavy"
        elif all(subjective.get(key) is not None for key in ("fatigue", "leg_soreness")):
            state = "acceptable" if max(subjective["fatigue"], subjective["leg_soreness"]) <= 3 else "heavy"
        else:
            state = "unknown"
        signals = (f"Subjective gate state: {state}.", reason)
    return OptionalGate(
        state, ("fresh", "acceptable"),
        {"sport": "other", "title": "Rest", "duration_seconds": 0}, signals,
    )


def _follows_key_session(session, sessions, config=None) -> bool:
    day = date.fromisoformat(session["planned_date"])
    return any(
        date.fromisoformat(item["planned_date"]) == day - timedelta(days=1)
        and _priority(item, config)[0] == "A" for item in sessions
    )


def _session_on_date(sessions, value: date) -> bool:
    return any(date.fromisoformat(item["planned_date"]) == value for item in sessions)


def _is_easy_session(session) -> bool:
    # Title and explicit intensity are more reliable than broad workbook row categories.
    primary = " ".join(str(session.get(key) or "") for key in ("title", "intensity")).lower()
    return any(re.search(rf"\b{re.escape(token)}\b", primary) for token in ("easy", "recovery", "z2", "aerobic")) and not any(
        re.search(rf"\b{re.escape(token)}\b", primary) for token in ("threshold", "vo2", "interval", "tempo")
    )


def _is_meaningful_stressor(cost: SessionCost) -> bool:
    return cost.systemic_cost == "high" or cost.muscular_cost == "high" or (
        cost.muscle_load == "lower" and cost.muscular_cost == "moderate"
    )


def _runway(latest, key_session, config) -> float | None:
    if not latest or not key_session:
        return None
    key_time = _day_start(date.fromisoformat(key_session["planned_date"]), config) + timedelta(hours=9)
    return max(0.0, (key_time - latest["started_at"]).total_seconds() / 3600)


def _day_start(value: date, config) -> datetime:
    zone = ZoneInfo(config.get("athlete", {}).get("timezone", "UTC"))
    return datetime.combine(value, time.min, zone).astimezone(timezone.utc)


def _runway_threshold(config, key, default) -> float:
    return float(config.get("coaching", {}).get("cost_model", {}).get("recovery_runway_hours", {}).get(key, default))


def _confidence(actual, subjective) -> str:
    if actual and all(item["cost"].confidence == "high" for item in actual):
        return "high"
    if actual:
        return "medium"
    return "low" if subjective is None else "medium"


def _candidate_actions(selected: str) -> list[dict[str, Any]]:
    order = ["KEEP", "REDUCE_DURATION", "REDUCE_INTENSITY", "REPLACE_WITH_EASY", "MOVE", "SKIP", "CONDITIONAL"]
    return [{"action": action, "selected": action == selected} for action in order]


def _actual_dict(item) -> dict[str, Any]:
    return {
        "activity_id": int(item["id"]), "source_activity_id": item.get("source_activity_id"),
        "date": item["started_at"].date().isoformat(), "started_at": item["started_at"].isoformat(),
        "sport": item["sport"], "name": item.get("name"),
        "duration_seconds": item.get("duration_seconds"), "session_cost": item["cost"].to_dict(),
    }


def _persist(item: Prescription, database) -> int:
    now = datetime.now(timezone.utc).isoformat()
    original_json = json.dumps(item.original, ensure_ascii=False, sort_keys=True)
    prescribed_json = json.dumps(item.prescribed, ensure_ascii=False, sort_keys=True)
    evidence_json = json.dumps(item.evidence, ensure_ascii=False, sort_keys=True)
    trace_json = json.dumps(item.decision_trace, ensure_ascii=False, sort_keys=True)
    gate_json = json.dumps(item.optional_gate.to_dict(), ensure_ascii=False) if item.optional_gate else None
    protected_json = json.dumps(item.protected_sessions, ensure_ascii=False)
    with connect(database) as connection:
        current = connection.execute(
            "SELECT * FROM workout_prescriptions WHERE planned_session_id=? AND status='active'",
            (item.planned_session_id,),
        ).fetchone()
        if current and current["action"] == item.action and current["decision_trace_json"] == trace_json:
            return int(current["id"])
        if current:
            connection.execute(
                "UPDATE workout_prescriptions SET status=? WHERE id=?",
                (f"superseded:{now}", current["id"]),
            )
        prescribed = item.prescribed or item.original or {}
        cursor = connection.execute(
            """INSERT INTO workout_prescriptions (
                   planned_session_id, prescribed_date, sport, title, description,
                   duration_seconds, distance_m, intensity, targets_json, reason,
                   status, action, confidence, original_json, prescribed_json,
                   evidence_json, rules_json, as_of_date, engine_version,
                   optional_gate_json, decision_trace_json, recovery_runway_h,
                   protected_sessions_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.planned_session_id, item.date.isoformat(), prescribed.get("sport", "other"),
                prescribed.get("title", "Skipped"), prescribed.get("description"),
                prescribed.get("duration_seconds"), prescribed.get("distance_m"), prescribed.get("intensity"),
                json.dumps(prescribed.get("targets", {}), ensure_ascii=False), " ".join(item.reasons),
                item.action, item.confidence, original_json, prescribed_json, evidence_json,
                json.dumps(item.rules_triggered, ensure_ascii=False), item.date.isoformat(), ENGINE_VERSION,
                gate_json, trace_json, item.recovery_runway_h, protected_json, now,
            ),
        )
        new_id = int(cursor.lastrowid)
        if current:
            connection.execute("UPDATE workout_prescriptions SET superseded_by=? WHERE id=?", (new_id, current["id"]))
        connection.commit()
    return new_id
