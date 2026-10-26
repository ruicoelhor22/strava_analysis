"""Lifecycle/provenance helpers on top of the existing imported plan tables."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone

from endurance_lab.db import connect, init_db


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def emit_event(connection, kind, payload, *, session_id=None, revision_id=None, key=None):
    connection.execute(
        "INSERT OR IGNORE INTO plan_events(event_type,created_at,planned_session_id,revision_id,payload_json,dedup_key) VALUES (?,?,?,?,?,?)",
        (kind, utc_now(), session_id, revision_id, json.dumps(payload, ensure_ascii=False, default=str), key),
    )


def infer_lifecycle(session):
    status = str(session.get('source_status') or '').strip().lower()
    title = str(session.get('title') or '').strip()
    description=str(session.get('description') or '')
    if status in {'skipped', 'cancelled', 'canceled'}:
        return 'cancelled', 'Coach explicitly cancelled this session.'
    actual=json.loads(session.get('workbook_actual_json') or '{}')
    if status in {'completed', 'modified'} and actual.get('Actual Session'):
        return 'completed', 'Coach recorded a completion; measured activity evidence remains separate.'
    if not title or title == 'Workout not finalized' or re.search(r'\b(tbd|tbc|placeholder|to be decided|to be confirmed)\b', title+' '+description, re.I):
        return 'incomplete', 'Workout content is missing or marked as a placeholder.'
    if not session.get('sport') or (not session.get('planned_duration_seconds') and not re.search(r'\b(rest|holiday|off|reset)\b', title, re.I)):
        return 'incomplete', 'Workout sport or duration is insufficiently specified.'
    raw=session.get('raw_source') or json.loads(session.get('raw_source_text') or '{}')
    if raw and not raw.get('Sport'):
        return 'incomplete', 'Coach has not specified the workout sport.'
    if not session.get('description') and not session.get('intensity'):
        return 'incomplete', 'No useful prescription or intensity is provided.'
    if status in {'planned', 'confirmed', 'approved', 'completed', 'modified'}:
        return 'confirmed', 'Explicit workbook scheduling status and usable workout details.'
    return 'draft', 'No explicit confirmation in the workbook; athlete confirmation is available.'


def phase_context(phase=None, week_type=None):
    original = str(phase or '')
    text = original.lower()
    # Week type is independent: a deload week inside BUILD remains BUILD + RECOVERY.
    def map_phase(value):
        if 'pre-holiday' in value or 'return from holiday' in value: return 'UNKNOWN'
        if re.search(r'\b(taper|polish|peak)\b', value): return 'TAPER'
        if re.search(r'\b(race|competition|event)\b', value): return 'RACE'
        if re.search(r'\b(recovery|deload|reset|holiday|transition)\b', value): return 'RECOVERY'
        if re.search(r'\b(build|threshold|specificity|durability|sharpen|vo2)\b', value): return 'BUILD'
        if re.search(r'\b(base|baseline|aerobic|foundation)\b', value): return 'BASE'
        return 'UNKNOWN'
    week_text = str(week_type or '').lower()
    week = 'RECOVERY' if re.search(r'deload|recovery|reset', week_text) else 'RACE' if 'race' in week_text else 'OVERLOAD' if 'overload' in week_text else 'TESTING' if re.search(r'test|validation', week_text) else 'NORMAL' if week_text else 'UNKNOWN'
    return {'phase': map_phase(text), 'phase_text': original, 'week_type': week, 'week_type_text': week_type}


def lifecycle_rows(database=None):
    with connect(database) as connection:
        return {int(r['planned_session_id']): dict(r) for r in connection.execute('SELECT * FROM plan_session_lifecycle')}


def session_workout(session):
    targets = session.get('targets', [])
    return {'date': session['planned_date'], 'sport': session['sport'], 'title': session['title'],
            'description': session.get('description'), 'duration_seconds': session.get('planned_duration_seconds'),
            'distance_m': session.get('planned_distance_m'), 'intensity': session.get('intensity'),
            'interval_structure': session.get('interval_structure'), 'priority': session.get('priority'),
            'targets': {t['target_type']: t['raw_text'] for t in targets}}


def resolve_session(session, prescription=None, decision=None, lifecycle=None):
    """Resolve presentation without turning an athlete choice into an app recommendation."""
    coach = session_workout(session)
    state = (lifecycle or {}).get('state', 'confirmed')  # legacy/manual records remain compatible
    recommendation = None
    if prescription and prescription.get('action') not in {'DRAFT', 'INCOMPLETE', 'CANCELLED', 'SKIP'}:
        recommendation = prescription.get('prescribed')
        if recommendation is None and prescription.get('prescribed_json'):
            recommendation = json.loads(prescription['prescribed_json'])
    effective = None
    source = 'APP PLAN' if session.get('source_sheet')=='App planning' else 'COACH'
    if decision:
        source = 'ME'
        if decision.get('action') != 'SKIP':
            effective = decision['workout']
    elif state in {'confirmed', 'completed'}:
        effective = recommendation or coach
        source = 'ENDURANCE LAB' if recommendation else source
        if prescription and prescription.get('action') == 'SKIP': effective = None
    return {'coach_plan': coach, 'app_recommendation': recommendation, 'athlete_decision': decision,
            'effective': effective, 'effective_source': source, 'lifecycle': state,
            'needs_review': bool(decision and decision.get('needs_review'))}


def revision_status(database=None):
    init_db(database)
    with connect(database) as connection:
        revision = connection.execute("SELECT * FROM plan_revisions WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()
        counts = Counter(r[0] for r in connection.execute('SELECT l.state FROM plan_session_lifecycle l JOIN planned_sessions p ON p.id=l.planned_session_id WHERE p.active=1'))
        decisions = connection.execute('SELECT count(*) FROM training_decisions').fetchone()[0]
        conflicts = connection.execute('SELECT count(*) FROM athlete_decision_meta WHERE needs_review=1').fetchone()[0]
    result = dict(revision) if revision else None
    if result:
        result['changes'] = json.loads(result['change_summary_json'])
        result['missing_dates'] = json.loads(result['missing_dates_json'])
    return {'revision': result, 'lifecycle_counts': dict(counts), 'athlete_decisions': decisions, 'decision_conflicts': conflicts}


def review_revision(revision_id, database=None):
    with connect(database) as connection:
        connection.execute('UPDATE plan_revisions SET reviewed_at=? WHERE id=?', (utc_now(), revision_id))
        connection.commit()


def confirm_session(session_id, database=None):
    init_db(database)
    with connect(database) as connection:
        row = connection.execute('SELECT * FROM plan_session_lifecycle WHERE planned_session_id=?', (session_id,)).fetchone()
        if not row or row['state'] != 'draft':
            raise ValueError('Only a sufficiently specified draft can be confirmed. Incomplete sessions need a coach revision or explicit modification.')
        connection.execute("UPDATE plan_session_lifecycle SET state='confirmed',reason='Explicit athlete confirmation',confirmed_at=? WHERE planned_session_id=?", (utc_now(), session_id))
        emit_event(connection, 'SESSION_CONFIRMED', {'state': 'confirmed'}, session_id=session_id, revision_id=row['revision_id'])
        connection.commit()


def session_history(session_id, database=None):
    with connect(database) as connection:
        versions = [dict(r) for r in connection.execute('SELECT v.*,r.imported_at FROM plan_session_versions v JOIN plan_revisions r ON r.id=v.revision_id WHERE planned_session_id=? ORDER BY r.imported_at', (session_id,))]
        decisions = [dict(r) for r in connection.execute('SELECT * FROM athlete_decision_history WHERE planned_session_id=? ORDER BY id', (session_id,))]
        prescriptions = [dict(r) for r in connection.execute('SELECT created_at,action,prescribed_json,status FROM workout_prescriptions WHERE planned_session_id=? ORDER BY id', (session_id,))]
    return {'coach_revisions': versions, 'athlete_decisions': decisions, 'app_prescriptions': prescriptions}
