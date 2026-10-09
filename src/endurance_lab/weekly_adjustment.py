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
from endurance_lab.training_decisions import effective_session, training_decisions, is_rest_session
from endurance_lab.plan_lifecycle import lifecycle_rows, phase_context, emit_event


ENGINE_VERSION = 4


def adjust_week(
    week_date: date,
    database: str | Path | None = None,
    *,
    as_of: date | None = None,
    persist: bool = True,
    respect_decisions: bool = True,
    strict_history: bool = False,
) -> list[Prescription]:
    """Reconcile actual cost against the remaining plan without future-data leakage."""
    init_db(database)
    config = load_athlete_config()
    as_of = as_of or date.today()
    monday = week_date - timedelta(days=week_date.weekday())
    sunday = monday + timedelta(days=6)
    known_at = (_day_start(as_of, config)).isoformat() if strict_history else None
    sessions = _planned_sessions(monday-timedelta(days=1), sunday+timedelta(days=3), database, known_at=known_at)
    actual = _actual_context(monday - timedelta(days=7), min(as_of, sunday + timedelta(days=1)), database, persist)
    subjective = latest_checkin(as_of, database)
    if known_at and subjective and max(subjective.get('created_at') or '',subjective.get('updated_at') or '') > known_at:
        subjective = None
    results: list[Prescription] = []
    for session in sessions:
        if not monday.isoformat()<=session['planned_date']<=sunday.isoformat(): continue
        result = _decide(session.get('original_plan') or session, sessions, actual, as_of, subjective, config)
        if persist:
            result = replace(result, prescription_id=_persist(result, database))
        decision = session.get('training_decision') if respect_decisions else None
        if decision:
            action = 'SKIP' if decision.get('action') == 'SKIP' else 'REST' if is_rest_session(session) else 'USER_CHOICE'
            trace = {**result.decision_trace, 'app_recommendation':result.prescribed,'app_action':result.action,
                     'athlete_decision':decision,'effective_source':'ME','decision_needs_review':bool(decision.get('needs_review'))}
            reasons = (decision['reason'], 'Your explicit choice is preserved; the independent app recommendation remains available.')
            if decision.get('needs_review'): reasons += ('Coach instructions changed materially; review your saved decision.',)
            result = replace(result, date=date.fromisoformat(decision['workout']['date']), prescribed=None if action=='SKIP' else dict(decision['workout']),action=action,decision_trace=trace,reasons=reasons,rules_triggered=result.rules_triggered+('athlete_decision_over_automatic_adjustment',))
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
    original = _workout(session.get("original_plan") or session)
    phase = phase_context(session.get('phase_name'),session.get('week_type'))
    priority, priority_source = _priority(session, config)
    preceding = [
        item for item in actual
        if item["started_at"].date() < planned_date
        and item["started_at"] >= _day_start(planned_date, config) - timedelta(days=7)
    ]
    same_day_actual = [item for item in actual if item["started_at"].date() == planned_date]
    future_keys = [
        item for item in sessions
        if item["planned_date"] > session["planned_date"] and _priority(item, config)[0] == "A" and _active_context(item)
    ]
    protected = tuple({
        "planned_session_id": int(item["id"]), "date": item["planned_date"],
        "title": item["title"], "sport": item["sport"], "priority": "A",
    } for item in future_keys)
    stressors = [item for item in preceding if _is_meaningful_stressor(item["cost"])]
    latest = max(stressors, key=lambda item: item["started_at"], default=None)
    # A key workout needs its own recovery spacing, not the runway to a later
    # key workout (which can falsely make tomorrow's quality look well rested).
    next_key = session if priority == "A" else (future_keys[0] if future_keys else None)
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
    lifecycle = session.get('lifecycle_state', 'confirmed')
    if lifecycle in {'draft','incomplete','cancelled'} and not completed:
        action = lifecycle.upper()
        reasons = [session.get('lifecycle_reason') or 'Coach workout is not confirmed; no firm prescription is generated.']
        rules = ['confirmed_plan_required']
    elif completed:
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
    elif max(fatigue, soreness if _strength_region(session) != "upper" else 0) >= subjective_threshold and not _is_easy_session(session):
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
            overlaps=[item for item in preceding if item['cost'].activity_id in {flag['activity_id'] for flag in interference if flag['type']=='lower_body_strength'}]
            moderate_with_rest=bool(overlaps) and all(item['cost'].muscular_cost=='moderate' and item['cost'].systemic_cost!='high' and
                any(is_rest_session(s) and item['started_at'].date()<date.fromisoformat(s['planned_date'])<planned_date for s in sessions)
                for item in overlaps)
            action = "REDUCE_INTENSITY" if moderate_with_rest else "REPLACE_WITH_EASY"
            gate = None if moderate_with_rest else _optional_gate(subjective, "Recent lower-body strength makes another gym session redundant.")
            reasons = (["Moderate local strength cost is separated from today's gym by an explicitly confirmed rest day.",
                        "Keep a gentle maintenance/deload session; omit demanding or sore-muscle work instead of automatically cancelling all gym training."]
                       if moderate_with_rest else ["Recent strength produced meaningful lower-body muscular load.",
                                                 "A second support-strength session inside the interference window adds limited value."])
            if protected:
                reasons.append(f"The smaller change protects {protected[0]['title']} ({protected[0]['date']}).")
            rules = ["lower_body_strength_interference", "modify_support_before_key", "minimum_necessary_adjustment"]
            if moderate_with_rest: rules.append('confirmed_rest_between_moderate_strength_sessions')
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
        elif same_modality_high and _can_move(session, sessions, config):
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

    if action == "KEEP" and not interference and not completed and planned_date >= as_of:
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

    if lifecycle in {'confirmed','completed'} and not completed and planned_date >= as_of:
        recovery_week = phase['phase']=='RECOVERY' or phase['week_type']=='RECOVERY'
        if recovery_week:
            rules.append('respect_planned_recovery_volume')
            reasons.append('This is a planned recovery/deload period: lower volume is intentional, not training debt. No volume is added.')
        if phase['phase']=='BASE':
            rules.append('base_consistency_over_extra_intensity')
            reasons.append('Base context prioritizes sustainable aerobic consistency; no extra intensity is introduced.')
            high_recent=sum(item['cost'].systemic_cost=='high' and (planned_date-item['started_at'].date()).days<=3 for item in preceding)
            if priority=='C' and not _is_easy_session(session) and high_recent>=2 and action=='KEEP':
                action='REDUCE_INTENSITY'
                reasons.append('Two recent high-cost sessions are enough quality stress for this base block; keep optional work easy rather than accumulate more intensity.')
                rules.append('base_limit_optional_intensity_accumulation')
        if phase['phase']=='BUILD' and priority=='A':
            rules.append('build_protect_quality_anchor')
            reasons.append('Build context protects the coach’s A-priority quality stimulus before optional support work.')
        if phase['phase']=='BUILD' and priority=='C' and action=='REPLACE_WITH_EASY' and future_keys:
            action='SKIP';gate=None
            reasons.append('During a build block, overlapping optional support work is omitted to protect the next A-priority stimulus; no replacement volume is invented.')
            rules.append('build_remove_interfering_optional_support')
        if phase['phase'] in {'TAPER','RACE'} or phase['week_type']=='RACE':
            rules.append('freshness_over_rolling_load')
            reasons.append('Taper/race context prioritizes freshness; declining rolling load is not a reason to add training.')
            near_key = any(0 <= (date.fromisoformat(key['planned_date']) - planned_date).days <= 3 for key in future_keys)
            if near_key and session['sport']=='strength' and _strength_region(session)!='upper' and priority!='A':
                action='SKIP';gate=None
                rules.append('avoid_leg_damage_before_event')
                reasons.append('Lower/full-body support strength within three days of the protected event adds avoidable muscular cost; omit it rather than replace it.')
        preceding_race = any(_is_race_session(s) and date.fromisoformat(s['planned_date'])==planned_date-timedelta(days=1) for s in sessions)
        if preceding_race and priority!='A' and action not in {'REST','SKIP'}:
            action='CONDITIONAL';gate=_optional_gate(subjective,'Post-race recovery governs support training.')
            rules.append('post_race_recovery_gate')
            reasons.append('This follows a race/event; proceed only when recovery is acceptable, otherwise rest.')

    if action == "KEEP" and priority == "A":
        rules.append("protect_key_session")
        reasons = ["This A-priority session remains a protected driver of the week's intended adaptation."]
        if runway is not None:
            reasons.append(f"Measured recovery runway is approximately {runway:.0f} hours.")
    if action == "KEEP" and _is_easy_session(session):
        rules.append("preserve_easy_day")
        reasons.append("No missed volume is added; the prescribed easy intensity remains easy.")

    if action == "KEEP" and _strength_region(session) == "upper" and future_keys and (date.fromisoformat(future_keys[0]["planned_date"]) - planned_date).days <= 1:
        rules.append("upper_body_maintenance_before_key")
        reasons.append("Keep upper-body work at maintenance effort; avoid failure, extra volume or added leg work before the key session.")
    prescribed = _adapt(original, action, planned_date, gate, config)
    if "upper_body_maintenance_before_key" in rules:
        prescribed["description"] = (str(prescribed.get("description") or "") + " " + reasons[-1]).strip()
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
        "phase_context": phase,
        "lifecycle": lifecycle,
        "planned_strength_region": _strength_region(session),
    }
    evidence = ({
        "session_costs": [item["cost"].to_dict() for item in preceding],
        "interference_flags": interference,
        "recovery_runway_h": runway,
        "priority": priority,
        "priority_source": priority_source,
    },)
    return Prescription(
        date.fromisoformat(prescribed['date']) if prescribed else planned_date, int(session["id"]), original, prescribed, action,
        tuple(reasons), evidence, trace["confidence"], tuple(rules), None,
        gate, trace, runway, protected,
    )


def _actual_context(start: date, end: date, database, persist: bool) -> list[dict[str, Any]]:
    config=load_athlete_config()
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT id, source_activity_id, name, sport, started_at,
                      COALESCE(moving_seconds, elapsed_seconds, 0) AS duration_seconds
               FROM activities WHERE started_at >= ? AND started_at < ? ORDER BY started_at""",
            (_day_start(start,config).isoformat(), _day_start(end,config).isoformat()),
        )]
    return [
        {**row, "started_at": datetime.fromisoformat(row["started_at"]),
         "cost": session_cost(int(row["id"]), database, persist=persist)}
        for row in rows
    ]


def _planned_sessions(start: date, end: date, database, *, known_at=None) -> list[dict[str, Any]]:
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT p.*, w.phase_name,w.week_type, a.id AS activity_id, a.name AS actual_name
               FROM planned_sessions p
               LEFT JOIN training_weeks w ON w.id=p.week_id
               LEFT JOIN planned_activity_matches m ON m.planned_session_id=p.id AND m.is_selected=1
               LEFT JOIN activities a ON a.id=m.activity_id
               WHERE p.active=1 OR EXISTS(SELECT 1 FROM training_decisions td WHERE td.planned_session_id=p.id)
               ORDER BY p.planned_date, p.id""",
            (),
        )]
        if known_at:
            legacy_rows = [row for row in rows if str(row.get('updated_at') or '9999')<=known_at and not connection.execute('SELECT 1 FROM plan_revisions WHERE plan_id=?',(row['plan_id'],)).fetchone()]
            revisions = connection.execute('SELECT id FROM plan_revisions r WHERE imported_at<=? AND id=(SELECT id FROM plan_revisions rr WHERE rr.plan_id=r.plan_id AND rr.imported_at<=? ORDER BY imported_at DESC,id DESC LIMIT 1)', (known_at,known_at)).fetchall()
            historical = []
            for rev in revisions:
                for version in connection.execute('SELECT * FROM plan_session_versions WHERE revision_id=?',(rev['id'],)):
                    snapshot = json.loads(version['snapshot_json'])
                    snapshot['id'] = version['planned_session_id']
                    confirmed=connection.execute("SELECT 1 FROM plan_events WHERE event_type='SESSION_CONFIRMED' AND planned_session_id=? AND revision_id=? AND created_at<=?",(snapshot['id'],version['revision_id'],known_at)).fetchone()
                    if confirmed: snapshot['lifecycle_state']='confirmed'
                    snapshot['activity_id'] = None
                    match = connection.execute('SELECT a.id,a.name FROM planned_activity_matches m JOIN activities a ON a.id=m.activity_id WHERE m.planned_session_id=? AND m.is_selected=1 AND a.started_at<?',(snapshot['id'],known_at)).fetchone()
                    if match: snapshot.update(activity_id=match['id'],actual_name=match['name'])
                    historical.append(snapshot)
            rows = historical + legacy_rows
        for row in rows:
            if known_at: continue
            row["targets"] = [dict(item) for item in connection.execute(
                "SELECT * FROM planned_session_targets WHERE planned_session_id=? ORDER BY target_type",
                (row["id"],),
            )]
    decisions = training_decisions(database,known_at=known_at)
    lifecycle = lifecycle_rows(database)
    results=[]
    for row in rows:
        if known_at and row.get('active')==0 and row['id'] not in decisions: continue
        raw=row.get('raw_source') or json.loads(row.get('raw_source_text') or '{}')
        row['phase_name']=raw.get('Phase') or row.get('phase_name')
        row['week_type']=raw.get('Week Type') or row.get('week_type')
        if not known_at and row['id'] in lifecycle:
            row.update(lifecycle_state=lifecycle[row['id']]['state'],lifecycle_reason=lifecycle[row['id']]['reason'])
        row = effective_session(row, decisions.get(int(row['id'])))
        if start.isoformat() <= row['planned_date'] <= end.isoformat(): results.append(row)
    return sorted(results,key=lambda r:(r['planned_date'],r['id']))


def _interference(session, preceding, config) -> list[dict[str, Any]]:
    planned_start = _day_start(date.fromisoformat(session["planned_date"]), config) + timedelta(hours=9)
    sport = str(session["sport"])
    flags = []
    for item in preceding:
        hours = (planned_start - item["started_at"]).total_seconds() / 3600
        if hours < 0 or hours > 96:
            continue
        cost = item["cost"]
        if cost.sport_specific_cost == "strength" and cost.muscle_load in {"lower", "full"} and cost.muscular_cost in {"moderate", "high"} and sport in {"cycling", "running", "strength"} and _strength_region(session) != "upper":
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
    if _is_race_session(session): return 'A', 'explicit_event'
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
    if action in {"SKIP", "DRAFT", "INCOMPLETE", "CANCELLED"}:
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
        result["duration_seconds"] = min(float(result['duration_seconds']), max(minimum, round(float(result["duration_seconds"]) * fraction / 300) * 300))
        result["title"] = f"Shortened: {result['title']}"
    elif action == "REDUCE_INTENSITY":
        result["title"] = f"Easy version: {result['title']}"
        result["description"] = "Keep this session conversational and remove quality intervals or hard targets."
        result["intensity"] = "Easy / recovery"
        result["interval_structure"] = None
        result["targets"] = {}
        if original.get("sport") == "strength":
            result["description"] = "Use comfortable maintenance loads, stop well short of failure and omit demanding or sore-muscle work. Do not add sets or chase personal records."
            result["intensity"] = "Easy strength / maintenance"
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
        if original.get("sport") == "strength":
            result.update(sport="other", title="Rest / recovery", duration_seconds=0,
                          intensity="Rest", description="Recover rather than adding an unplanned run after overlapping strength work.")
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
        and _priority(item, config)[0] == "A" and _active_context(item) for item in sessions
    )


def _session_on_date(sessions, value: date) -> bool:
    return any(date.fromisoformat(item["planned_date"]) == value and _active_context(item) for item in sessions)


def _active_context(session) -> bool:
    return not session.get('decision_skipped') and not is_rest_session(session) and session.get('lifecycle_state') not in {'draft','incomplete','cancelled'}


def _strength_region(session) -> str | None:
    if session.get("sport") != "strength":
        return None
    title = str(session.get("title") or "").lower()
    lower = bool(re.search(r"\b(legs?|lower|squats?|deadlift|rdl)\b", title))
    upper = bool(re.search(r"\b(upper|pull|push|arms?|chest|back|shoulders?)\b", title))
    return "full" if lower and upper else "lower" if lower else "upper" if upper else "unknown"


def _can_move(session, sessions, config) -> bool:
    target = date.fromisoformat(session["planned_date"]) + timedelta(days=1)
    if _session_on_date(sessions, target):
        return False
    return not any(_priority(item, config)[0] == "A" and
                   0 <= (date.fromisoformat(item["planned_date"]) - target).days < 2
                   for item in sessions if item["id"] != session["id"] and _active_context(item))


def _is_easy_session(session) -> bool:
    # Title and explicit intensity are more reliable than broad workbook row categories.
    primary = " ".join(str(session.get(key) or "") for key in ("title", "intensity")).lower()
    return any(re.search(rf"\b{re.escape(token)}\b", primary) for token in ("easy", "recovery", "z2", "aerobic")) and not any(
        re.search(rf"\b{re.escape(token)}\b", primary) for token in ("threshold", "vo2", "interval", "tempo")
    )


def _is_race_session(session):
    return bool(re.search(r'\b(race|competition|event day)\b', str(session.get('title') or ''), re.I))


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
    order = ["KEEP", "REDUCE_DURATION", "REDUCE_INTENSITY", "REPLACE_WITH_EASY", "MOVE", "SKIP", "CONDITIONAL", "REST", "USER_CHOICE"]
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
        emit_event(connection,'PRESCRIPTION_CHANGED',{'action':item.action,'prescription_id':new_id},session_id=item.planned_session_id,key=f'prescription:{new_id}')
        if current:
            connection.execute("UPDATE workout_prescriptions SET superseded_by=? WHERE id=?", (new_id, current["id"]))
        connection.commit()
    return new_id
