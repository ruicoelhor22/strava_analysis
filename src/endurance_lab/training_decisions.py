"""Explicit athlete decisions that survive automatic coaching reconciliation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from endurance_lab.db import connect, init_db
from endurance_lab.plan_lifecycle import emit_event, utc_now, session_workout


def save_training_decision(session_id: int, workout: dict, reason: str, database=None, *, action='MODIFY', source='athlete') -> None:
    init_db(database)
    if not reason.strip():
        raise ValueError("A training decision needs a reason")
    with connect(database) as connection:
        session = connection.execute("SELECT * FROM planned_sessions WHERE id=?", (session_id,)).fetchone()
        if session is None:
            raise ValueError("Active planned session not found")
        if not session['active'] and not connection.execute('SELECT 1 FROM training_decisions WHERE planned_session_id=?',(session_id,)).fetchone():
            raise ValueError('Coach removed this session; no existing choice is available to retain')
        from datetime import date
        date.fromisoformat(str(workout.get('date')))
        if action not in {'ACCEPT', 'KEEP_ORIGINAL', 'MODIFY', 'MOVE', 'SKIP', 'CONDITIONAL'}:
            raise ValueError('Unsupported athlete decision action')
        if not workout.get("title") or not workout.get("sport"):
            raise ValueError("A training decision needs a sport and title")
        duration=float(workout.get('duration_seconds') or 0)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Duration cannot be negative")
        if workout['sport'] not in {'cycling','running','swimming','strength','other'}:
            raise ValueError('Unsupported workout sport')
        if connection.execute(
            "SELECT 1 FROM planned_activity_matches WHERE planned_session_id=? AND is_selected=1", (session_id,)
        ).fetchone():
            raise ValueError("Cannot replace an already matched completed session")
        if str(session['source_status'] or '').lower() in {'completed','modified'} and json.loads(session['workbook_actual_json'] or '{}').get('Actual Session'):
            raise ValueError('Cannot replace a completed workbook session')
        connection.execute(
            """INSERT INTO training_decisions(planned_session_id,workout_json,reason,created_at)
               VALUES (?,?,?,?) ON CONFLICT(planned_session_id) DO UPDATE SET
               workout_json=excluded.workout_json, reason=excluded.reason""",
            (session_id, json.dumps(workout, ensure_ascii=False), reason.strip(), datetime.now(timezone.utc).isoformat()),
        )
        previous = training_decisions(database).get(session_id)
        revision = connection.execute('SELECT revision_id FROM plan_session_lifecycle WHERE planned_session_id=?',(session_id,)).fetchone()
        revision_id = revision[0] if revision else None
        now = utc_now()
        connection.execute('INSERT INTO athlete_decision_meta(planned_session_id,action,source,updated_at,based_revision_id,needs_review) VALUES (?,?,?,?,?,0) ON CONFLICT(planned_session_id) DO UPDATE SET action=excluded.action,source=excluded.source,updated_at=excluded.updated_at,based_revision_id=excluded.based_revision_id,needs_review=0', (session_id,action,source,now,revision_id))
        resulting = {'workout':workout,'reason':reason,'action':action,'created_at':now,'needs_review':0,'based_revision_id':revision_id}
        prior_prescription=connection.execute("SELECT prescribed_json FROM workout_prescriptions WHERE planned_session_id=? AND status='active'",(session_id,)).fetchone()
        previous_effective=(previous or {}).get('workout') or (json.loads(prior_prescription[0]) if prior_prescription and prior_prescription[0] else session_workout(dict(session)))
        resulting['previous_effective']=previous_effective
        resulting['resulting_effective']=None if action=='SKIP' else workout
        connection.execute('INSERT INTO athlete_decision_history(planned_session_id,created_at,action,source,previous_json,resulting_json,note,based_revision_id) VALUES (?,?,?,?,?,?,?,?)', (session_id,now,action,source,json.dumps(previous,ensure_ascii=False),json.dumps(resulting,ensure_ascii=False),reason,revision_id))
        emit_event(connection,'ATHLETE_DECISION_CHANGED',{'action':action},session_id=session_id,revision_id=revision_id)
        connection.commit()


def training_decisions(database=None, *, known_at=None) -> dict[int, dict]:
    with connect(database) as connection:
        if known_at is not None:
            rows = connection.execute('SELECT * FROM athlete_decision_history WHERE created_at<=? ORDER BY created_at,id',(known_at,)).fetchall()
            results = {}
            for row in rows:
                if row['action']=='CLEAR': results.pop(row['planned_session_id'],None)
                else: results[row['planned_session_id']] = json.loads(row['resulting_json'])
            return results
        return {int(row["planned_session_id"]): {**dict(row), "workout": json.loads(row["workout_json"])}
                for row in connection.execute("SELECT d.*,COALESCE(m.action,'MODIFY') AS action,m.updated_at,m.based_revision_id,COALESCE(m.needs_review,0) AS needs_review FROM training_decisions d LEFT JOIN athlete_decision_meta m ON m.planned_session_id=d.planned_session_id")}


def clear_training_decision(session_id, database=None):
    init_db(database)
    previous = training_decisions(database).get(session_id)
    if not previous: return
    with connect(database) as connection:
        connection.execute("INSERT INTO athlete_decision_history(planned_session_id,created_at,action,source,previous_json,note) VALUES (?,?,'CLEAR','athlete',?,'Athlete cleared the explicit choice')", (session_id,utc_now(),json.dumps(previous,ensure_ascii=False)))
        connection.execute('DELETE FROM training_decisions WHERE planned_session_id=?',(session_id,))
        connection.execute('DELETE FROM athlete_decision_meta WHERE planned_session_id=?',(session_id,))
        connection.commit()


def choose_training(session_id, action, database=None, *, workout=None, note=None, replace_decision=False):
    init_db(database)
    if action=='CLEAR':
        clear_training_decision(session_id,database); return
    existing = training_decisions(database).get(session_id)
    if action=='RETAIN':
        if not existing: raise ValueError('No saved decision to keep')
        save_training_decision(session_id,existing['workout'],note or existing['reason'],database,
                               action=existing['action'])
        return
    with connect(database) as connection:
        row = connection.execute('SELECT * FROM planned_sessions WHERE id=?',(session_id,)).fetchone()
        if not row: raise ValueError('Active planned session not found')
        session = dict(row)
        session['targets'] = [dict(t) for t in connection.execute('SELECT * FROM planned_session_targets WHERE planned_session_id=?',(session_id,))]
    if action=='ACCEPT':
        # Accept the independent recommendation displayed by the UI, not a new
        # historical-day calculation or the athlete-resolved Today card.
        with connect(database) as connection:
            recommendation = connection.execute("SELECT action,prescribed_json FROM workout_prescriptions WHERE planned_session_id=? AND status='active'",(session_id,)).fetchone()
        if not recommendation:
            from endurance_lab.weekly_adjustment import adjust_week
            from datetime import date
            adjust_week(date.fromisoformat(session['planned_date']),database,persist=True,respect_decisions=False)
            with connect(database) as connection:
                recommendation = connection.execute("SELECT action,prescribed_json FROM workout_prescriptions WHERE planned_session_id=? AND status='active'",(session_id,)).fetchone()
        if not recommendation or recommendation['action'] in {'DRAFT','INCOMPLETE','CANCELLED'}:
            raise ValueError('No confirmed app recommendation to accept')
        if recommendation['action']=='SKIP':
            action='SKIP';workout=session_workout(session)
        else:
            recommended = json.loads(recommendation['prescribed_json'] or 'null')
            if not recommended: raise ValueError('No confirmed app recommendation to accept')
            if workout is not None and workout != recommended:
                raise ValueError('The app recommendation changed. Review it again before accepting.')
            workout = recommended
        if existing and (existing['workout'] != workout or existing['action']=='SKIP' or action=='SKIP') and not replace_decision:
            raise ValueError('Accepting this app recommendation would replace your saved decision. Confirm replacement, or keep your decision.')
    elif action=='KEEP_ORIGINAL':
        if not session['active']: raise ValueError('Coach removed this workout. Clear your decision to follow that removal.')
        from endurance_lab.plan_lifecycle import lifecycle_rows
        if lifecycle_rows(database).get(session_id,{}).get('state') in {'incomplete','cancelled'}:
            raise ValueError('This coach entry is incomplete or cancelled. Modify it explicitly, or choose Skip.')
        workout = session_workout(session)
    elif action=='SKIP':
        workout = session_workout(session)
    elif workout is None:
        raise ValueError('A modification or move needs a workout')
    save_training_decision(session_id,workout,note or action.replace('_',' ').title(),database,action=action)


def effective_session(session: dict, decision: dict | None) -> dict:
    """Project a decision without rewriting the imported workbook's original plan."""
    if decision is None:
        return dict(session)
    workout = decision["workout"]
    result = {**dict(session), "original_plan": dict(session), "training_decision": decision}
    result['planned_date'] = workout.get('date', session['planned_date'])
    for field in ("sport", "title", "description", "intensity", "interval_structure", "priority"):
        result[field] = workout.get(field)
    result["planned_duration_seconds"] = workout.get("duration_seconds", 0)
    result["planned_distance_m"] = workout.get("distance_m")
    result["planned_load"] = None
    result["session_type"] = workout.get("intensity")
    result["targets"] = []
    result["workbook_actual_json"] = "{}"
    result["source_status"] = None
    if decision.get('action') == 'SKIP':
        result['planned_duration_seconds'] = 0
        result['decision_skipped'] = True
    return result


def is_rest_session(session: dict) -> bool:
    return bool(session.get("training_decision")) and session.get("sport") == "other" and not session.get("planned_duration_seconds")
