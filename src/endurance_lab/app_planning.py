"""Explicitly requested app plans, separate from imported coach workbooks."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from endurance_lab.db import connect, init_db
from endurance_lab.plan_lifecycle import emit_event, utc_now


def save_app_plan(source, database=None):
    """Add a dated proposal once; never overwrite existing coach or completed work."""
    source = Path(source).resolve()
    content = source.read_bytes()
    plan = json.loads(content)
    sessions = plan['sessions']
    start, end = date.fromisoformat(plan['date_start']), date.fromisoformat(plan['date_end'])
    days = [s['date'] for s in sessions]
    if len(set(days)) != len(days) or set(days) != {
        (start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)
    }:
        raise ValueError('Specify exactly one workout or rest entry per proposal day')
    for session in sessions:
        if session['sport'] not in {'cycling','running','strength','other'} or not session['title']:
            raise ValueError('Unsupported or incomplete session')
        duration = float(session['minutes'])
        if not 0 <= duration <= 1440:
            raise ValueError('Invalid duration')
        for restriction in plan.get('restrictions', []):
            if restriction['start'] <= session['date'] <= restriction['end']:
                rest = session['sport']=='other' and duration==0
                if not rest and session['sport'] not in restriction['allowed_sports']:
                    raise ValueError('Session violates travel availability')
    digest = hashlib.sha256(content).hexdigest()
    init_db(database)
    now = utc_now()
    with connect(database) as c:
        existing = c.execute('SELECT id,source_sha256 FROM training_plans WHERE source_path=?',(str(source),)).fetchone()
        if existing:
            if existing['source_sha256'] != digest:
                raise ValueError('Existing proposal changed; review it as a new revision instead of overwriting')
            return [r[0] for r in c.execute('SELECT id FROM planned_sessions WHERE plan_id=? ORDER BY planned_date',(existing['id'],))]
        if c.execute('SELECT 1 FROM planned_sessions WHERE active=1 AND planned_date BETWEEN ? AND ? LIMIT 1',(str(start),str(end))).fetchone():
            raise ValueError('Existing sessions overlap this proposal; review before replacing them')
        pid = c.execute('''INSERT INTO training_plans(name,source_path,source_filename,source_sha256,imported_at,date_start,date_end,metadata_json)
                           VALUES (?,?,?,?,?,?,?,?)''',
                        (plan['name'],str(source),source.name,digest,now,str(start),str(end),json.dumps({'source_kind':'app_proposal','restrictions':plan.get('restrictions',[])}))).lastrowid
        revision = c.execute('''INSERT INTO plan_revisions(plan_id,imported_at,source_filename,source_hash,date_start,date_end,session_count,reviewed_at)
                                VALUES (?,?,?,?,?,?,?,?)''',(pid,now,source.name,digest,str(start),str(end),len(sessions),now)).lastrowid
        weeks = {}
        ids = []
        for index, session in enumerate(sessions):
            day = date.fromisoformat(session['date'])
            monday = day-timedelta(days=day.weekday())
            if monday not in weeks:
                weeks[monday] = c.execute('''INSERT INTO training_weeks(plan_id,week_start,week_end,phase_name,week_type,objectives,raw_json)
                                              VALUES (?,?,?,?,?,?,?)''',
                    (pid,str(monday),str(monday+timedelta(days=6)),plan['name'],'Maintenance',plan['rationale'],json.dumps({'source_kind':'app_proposal'}))).lastrowid
            sid = c.execute('''INSERT INTO planned_sessions(plan_id,week_id,planned_date,sport,title,description,planned_duration_seconds,
                intensity,interval_structure,priority,priority_source,notes,source_status,workbook_actual_json,source_sheet,source_row,
                source_key,raw_source_text,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (pid,weeks[monday],str(day),session['sport'],session['title'],session['description'],float(session['minutes'])*60,
                 session['intensity'],session.get('structure'),session['priority'],'app_proposal',
                 'App-generated proposal requested by athlete; not a finalized human-coach prescription.',
                 'planned','{}','App planning',index+1,'app:'+str(day),json.dumps({'source_kind':'app_proposal','restriction_context':plan.get('restrictions',[])}),now)).lastrowid
            for kind, text in session.get('targets',{}).items():
                c.execute('INSERT INTO planned_session_targets(planned_session_id,target_type,raw_text,confidence) VALUES (?,?,?,?)',(sid,kind,text,'high'))
            c.execute("INSERT INTO plan_session_lifecycle(planned_session_id,revision_id,state,reason,confirmed_at) VALUES (?,?,'confirmed',?,?)",
                      (sid,revision,'Athlete explicitly requested app planning for these dates; not a coach workbook session.',now))
            snapshot = dict(c.execute('SELECT * FROM planned_sessions WHERE id=?',(sid,)).fetchone())
            snapshot.update(phase_name=plan['name'],week_type='Maintenance',lifecycle_state='confirmed',targets=[dict(r) for r in c.execute('SELECT * FROM planned_session_targets WHERE planned_session_id=?',(sid,))])
            c.execute('INSERT INTO plan_session_versions VALUES (?,?,?)',(revision,sid,json.dumps(snapshot)))
            emit_event(c,'APP_PLAN_SESSION_ADDED',{'date':str(day),'source_kind':'app_proposal'},session_id=sid,revision_id=revision)
            ids.append(sid)
        emit_event(c,'APP_PLAN_CREATED',{'source_kind':'app_proposal','restrictions':plan.get('restrictions',[]),'rationale':plan['rationale']},revision_id=revision)
        c.commit()
    return ids
