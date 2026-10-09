"""Conservative identity matching and atomic revision metadata for workbook imports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import date, timedelta
from difflib import SequenceMatcher

from endurance_lab.db import connect, init_db
from endurance_lab.plan_lifecycle import emit_event, infer_lifecycle, utc_now
from endurance_lab.plan_parser import discover_workbook


FIELD_CHANGES = {
    'planned_date': 'SESSION_MOVED', 'sport': 'SESSION_TYPE_CHANGED',
    'planned_duration_seconds': 'DURATION_CHANGED', 'intensity': 'INTENSITY_CHANGED',
    'interval_structure': 'INTENSITY_CHANGED', 'priority': 'PRIORITY_CHANGED',
    'description': 'NOTES_CHANGED', 'notes': 'NOTES_CHANGED', 'title': 'NOTES_CHANGED',
    'phase_name': 'PHASE_CHANGED', 'week_type': 'WEEK_TYPE_CHANGED', 'source_status': 'STATUS_CHANGED',
}


def session_snapshot(session):
    row = asdict(session)
    row['planned_date'] = session.planned_date.isoformat()
    row['week_start'] = session.week_start.isoformat()
    row['phase_name'] = row.pop('phase')
    row['workbook_actual_json'] = json.dumps(row.pop('workbook_actual'), ensure_ascii=False)
    return row


def existing_sessions(connection, plan_id):
    rows=[dict(r) for r in connection.execute(
        """SELECT p.*, w.phase_name,w.week_type,
                  EXISTS(SELECT 1 FROM planned_activity_matches m WHERE m.planned_session_id=p.id AND m.is_selected=1) AS has_match
           FROM planned_sessions p LEFT JOIN training_weeks w ON w.id=p.week_id WHERE p.plan_id=? AND p.active=1""", (plan_id,))]
    for row in rows:
        raw=json.loads(row.get('raw_source_text') or '{}')
        row['phase_name']=raw.get('Phase') or row.get('phase_name')
        row['week_type']=raw.get('Week Type') or row.get('week_type')
        row['has_completion']=bool(row['has_match']) or (row['planned_date']<date.today().isoformat() and str(row.get('source_status') or '').lower() in {'completed','modified'} and bool(json.loads(row.get('workbook_actual_json') or '{}').get('Actual Session')))
    return rows


def bootstrap_revision(connection, plan_id):
    if connection.execute('SELECT 1 FROM plan_revisions WHERE plan_id=?', (plan_id,)).fetchone(): return
    plan = connection.execute('SELECT * FROM training_plans WHERE id=?', (plan_id,)).fetchone()
    rows = existing_sessions(connection, plan_id)
    revision_id = connection.execute(
        'INSERT INTO plan_revisions(plan_id,imported_at,source_filename,source_hash,date_start,date_end,session_count,reviewed_at) VALUES (?,?,?,?,?,?,?,?)',
        (plan_id, plan['imported_at'], plan['source_filename'], plan['source_sha256'], plan['date_start'], plan['date_end'], len(rows), utc_now()),
    ).lastrowid
    for row in rows:
        state, reason = infer_lifecycle(row)
        row['lifecycle_state'] = state
        connection.execute('INSERT INTO plan_session_versions VALUES (?,?,?)', (revision_id,row['id'],json.dumps(row,default=str)))
        connection.execute('INSERT OR IGNORE INTO plan_session_lifecycle(planned_session_id,revision_id,state,reason) VALUES (?,?,?,?)', (row['id'],revision_id,state,reason))
    populated={row['planned_date'] for row in rows}
    cursor=date.fromisoformat(plan['date_start']);missing=[]
    while cursor<=date.fromisoformat(plan['date_end']):
        if cursor.isoformat() not in populated: missing.append(cursor.isoformat())
        cursor+=timedelta(days=1)
    connection.execute('UPDATE plan_revisions SET missing_dates_json=? WHERE id=?',(json.dumps(missing),revision_id))
    for decision in connection.execute('SELECT * FROM training_decisions WHERE planned_session_id IN (SELECT id FROM planned_sessions WHERE plan_id=?)', (plan_id,)).fetchall():
        connection.execute('INSERT OR IGNORE INTO athlete_decision_meta(planned_session_id,updated_at,based_revision_id) VALUES (?,?,?)', (decision['planned_session_id'],decision['created_at'],revision_id))
        connection.execute("INSERT INTO athlete_decision_history(planned_session_id,created_at,action,source,resulting_json,note,based_revision_id) VALUES (?,?,'MODIFY','legacy',?,?,?)", (decision['planned_session_id'],decision['created_at'],json.dumps({**dict(decision),'action':'MODIFY','workout':json.loads(decision['workout_json'])}),decision['reason'],revision_id))


def compare_sessions(old, new):
    """Only accept mutual unique matches; ambiguous edits remain explicit add/remove."""
    assigned = set()
    matches = {}
    for index, item in enumerate(new):
        exact = [r for r in old if r['source_key'] == item['source_key']]
        if len(exact) == 1:
            matches[index] = exact[0]; assigned.add(exact[0]['id'])
    # Codes, date/sport edits, and near-identical titles can preserve identity.
    for index, item in enumerate(new):
        if index in matches: continue
        candidates = []
        for previous in old:
            if previous['id'] in assigned: continue
            days = abs((date.fromisoformat(previous['planned_date']) - date.fromisoformat(item['planned_date'])).days)
            code = item.get('workout_code')
            title_similarity = SequenceMatcher(None, str(previous['title']).lower(), str(item['title']).lower()).ratio()
            if (code and code == previous.get('workout_code') and days <= 7) or (days == 0 and item['sport'] == previous['sport']) or (days <= 1 and item['sport'] == previous['sport'] and title_similarity >= .80):
                candidates.append(previous)
        if len(candidates) != 1: continue
        previous = candidates[0]
        competitors = [j for j, other in enumerate(new) if j not in matches and j != index and
                       other['sport'] == previous['sport'] and other['planned_date'] == previous['planned_date']]
        if competitors: continue
        matches[index] = previous; assigned.add(previous['id'])
    changes = []
    for index,item in enumerate(new):
        previous = matches.get(index)
        if not previous:
            changes.append({'type':'SESSION_ADDED','date':item['planned_date'],'title':item['title'],'material':True})
            continue
        for field,kind in FIELD_CHANGES.items():
            before, after = previous.get(field), item.get(field)
            if (before or None) == (after or None): continue
            typo = field == 'title' and SequenceMatcher(None, str(before).lower(), str(after).lower()).ratio() >= .85
            # Wording-only notes do not invalidate decisions; actual interval/target changes do.
            material = field not in {'notes','description','title'} or (field == 'title' and not typo)
            changes.append({'type':kind,'field':field,'session_id':previous['id'],'date':item['planned_date'],'title':item['title'],
                            'before':before,'after':after,'material':material,'completed_match_preserved':bool(previous.get('has_match'))})
        before_targets = previous.get('targets', [])
        if before_targets != item.get('targets', []):
            changes.append({'type':'INTENSITY_CHANGED','field':'targets','session_id':previous['id'],'date':item['planned_date'],'title':item['title'],'material':True})
    for previous in old:
        if previous['id'] not in assigned:
            changes.append({'type':'SESSION_REMOVED','session_id':previous['id'],'date':previous['planned_date'],'title':previous['title'],'material':True,'completed_match_preserved':bool(previous.get('has_completion'))})
    return matches, changes


def prepare_revision(connection, parsed, plan_id):
    old = existing_sessions(connection, plan_id)
    for row in old:
        row['targets'] = [dict(t) for t in connection.execute('SELECT target_type,raw_text,minimum_value,maximum_value,unit,confidence FROM planned_session_targets WHERE planned_session_id=? ORDER BY target_type', (row['id'],))]
    new = [session_snapshot(s) for s in parsed.sessions]
    for row in new: row['targets'] = sorted(row['targets'],key=lambda t:t['target_type'])
    matches, changes = compare_sessions(old, new)
    sessions = tuple(replace(s, source_key=matches[i]['source_key']) if i in matches else s for i,s in enumerate(parsed.sessions))
    return replace(parsed,sessions=sessions), matches, changes


def finish_revision(connection, parsed, plan_id, changes, missing_dates):
    connection.execute("UPDATE plan_revisions SET status='superseded' WHERE plan_id=? AND status='active'", (plan_id,))
    revision_id = connection.execute(
        'INSERT INTO plan_revisions(plan_id,imported_at,source_filename,source_hash,date_start,date_end,session_count,change_summary_json,missing_dates_json) VALUES (?,?,?,?,?,?,?,?,?)',
        (plan_id,utc_now(),parsed.source.name,parsed.source_sha256,str(parsed.date_start),str(parsed.date_end),len(parsed.sessions),json.dumps(changes,ensure_ascii=False),json.dumps(missing_dates)),
    ).lastrowid
    material_ids = {c['session_id'] for c in changes if c.get('session_id') and c['material']}
    for change in changes:
        if change['type']=='SESSION_REMOVED' and not change.get('completed_match_preserved'):
            sid=change['session_id']
            connection.execute("UPDATE plan_session_lifecycle SET state='cancelled',reason='Coach removed this session in the latest revision',revision_id=? WHERE planned_session_id=?",(revision_id,sid))
            previous=connection.execute('SELECT * FROM planned_sessions WHERE id=?',(sid,)).fetchone()
            snapshot={**dict(previous),'active':0,'lifecycle_state':'cancelled'}
            connection.execute('INSERT INTO plan_session_versions VALUES (?,?,?)',(revision_id,sid,json.dumps(snapshot,default=str)))
    for session in parsed.sessions:
        row = connection.execute('SELECT * FROM planned_sessions WHERE plan_id=? AND source_key=?', (plan_id,session.source_key)).fetchone()
        snapshot = session_snapshot(session)
        state,reason = infer_lifecycle(snapshot)
        snapshot['id'] = row['id']; snapshot['lifecycle_state'] = state
        previous = connection.execute('SELECT * FROM plan_session_lifecycle WHERE planned_session_id=?', (row['id'],)).fetchone()
        if previous and previous['confirmed_at'] and row['id'] not in material_ids:
            state='confirmed'; reason='Athlete confirmation preserved across non-material revision.'
        snapshot['lifecycle_state']=state
        connection.execute('INSERT INTO plan_session_versions VALUES (?,?,?)', (revision_id,row['id'],json.dumps(snapshot,ensure_ascii=False,default=str)))
        connection.execute('INSERT INTO plan_session_lifecycle(planned_session_id,revision_id,state,reason) VALUES (?,?,?,?) ON CONFLICT(planned_session_id) DO UPDATE SET revision_id=excluded.revision_id,state=excluded.state,reason=excluded.reason,confirmed_at=CASE WHEN ? THEN NULL ELSE confirmed_at END', (row['id'],revision_id,state,reason,row['id'] in material_ids))
    for session_id in material_ids:
        if not connection.execute('SELECT 1 FROM planned_activity_matches WHERE planned_session_id=? AND is_selected=1',(session_id,)).fetchone():
            connection.execute("UPDATE workout_prescriptions SET status=? WHERE planned_session_id=? AND status='active'",(f'superseded:plan-revision:{revision_id}',session_id))
        if connection.execute('SELECT 1 FROM training_decisions WHERE planned_session_id=?',(session_id,)).fetchone():
            connection.execute('INSERT INTO athlete_decision_meta(planned_session_id,updated_at,based_revision_id,needs_review) VALUES (?,?,?,1) ON CONFLICT(planned_session_id) DO UPDATE SET needs_review=1', (session_id,utc_now(),revision_id))
            emit_event(connection,'PLAN_CONFLICT',{'reason':'Coach instruction changed materially. Saved choice preserved pending review.'},session_id=session_id,revision_id=revision_id)
    emit_event(connection,'PLAN_REVISION_IMPORTED',{'changes':len(changes),'sessions':len(parsed.sessions)},revision_id=revision_id)
    return revision_id


def sync_plan_revision(database=None, *, dry_run=False):
    """Failure is contained: previous valid plan remains usable by the rest of sync."""
    init_db(database)
    with connect(database) as connection:
        plan = connection.execute('SELECT * FROM training_plans WHERE active=1 ORDER BY imported_at DESC LIMIT 1').fetchone()
    try:
        source = discover_workbook(plan['source_path'] if plan else None)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if plan and digest == plan['source_sha256']:
            if not dry_run:
                with connect(database) as connection:
                    bootstrap_revision(connection,plan['id']); connection.commit()
            return {'status':'unchanged','changed':False}
        from endurance_lab.plan_importer import import_training_plan
        summary = import_training_plan(source,database,dry_run=dry_run)
        return {'status':'preview' if dry_run else 'imported','changed':True,'sessions':summary.sessions,'changes':list(summary.changes)}
    except Exception as exc:
        if not plan and isinstance(exc,(FileNotFoundError,ValueError)):
            return {'status':'not_configured','changed':False}
        if not dry_run:
            with connect(database) as connection:
                emit_event(connection,'PLAN_IMPORT_FAILED',{'error':str(exc),'existing_plan_preserved':True});connection.commit()
        return {'status':'failed','changed':False,'existing_plan_preserved':True,'error':str(exc)}
