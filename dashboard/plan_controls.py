"""Compact athlete controls shared by Today, Plan and Coach."""
from datetime import date, timedelta
import json

import streamlit as st

from dashboard._shared import format_duration
from endurance_lab.db import connect
from endurance_lab.plan_lifecycle import (
    resolve_session, lifecycle_rows, revision_status, review_revision, confirm_session, session_history,
)
from endurance_lab.training_decisions import training_decisions, choose_training


def _refresh():
    from endurance_lab.plan_matching import match_training_plan
    from endurance_lab.weekly_adjustment import adjust_week
    from endurance_lab.google_calendar import sync_google_calendar, calendar_enabled
    match_training_plan()
    for offset in (0,7,14): adjust_week(date.today()+timedelta(days=offset))
    try:
        if calendar_enabled():
            result=sync_google_calendar()
            if result.errors: st.warning('Choice saved locally; calendar sync reported errors and will retry during normal sync.')
    except Exception:
        st.warning('Choice saved locally. Calendar update is pending; normal sync will retry.')
    st.cache_data.clear()


def _label(workout):
    if not workout: return 'No confirmed workout'
    return f"{workout.get('date','')} · {workout.get('title','')} · {format_duration(workout.get('duration_seconds'))}"


def revision_panel():
    status=revision_status()
    revision=status['revision']
    with connect() as connection:
        failure=connection.execute("SELECT created_at,payload_json FROM plan_events WHERE event_type='PLAN_IMPORT_FAILED' ORDER BY id DESC LIMIT 1").fetchone()
    if failure and (not revision or failure['created_at']>revision['imported_at']):
        st.warning('Plan import failed. The previous valid plan is still active. Check the workbook and retry sync.')
    if not revision: return
    if status['decision_conflicts']: st.warning(f"{status['decision_conflicts']} saved decision(s) need review after a coach revision.")
    with connect() as connection:
        conflicts=[r[0] for r in connection.execute('SELECT planned_session_id FROM athlete_decision_meta WHERE needs_review=1')]
    for sid in conflicts:
        session_controls(sid,key_prefix='revision-conflict')
    with st.expander('Plan updated · Review changes' if not revision['reviewed_at'] else 'Plan revision history', expanded=False):
        st.caption(f"Revision {revision['id']} · {revision['imported_at'][:16].replace('T',' ')} UTC · {len(revision['changes'])} changes")
        st.write('Session states:', status['lifecycle_counts'])
        if revision['missing_dates']:
            st.caption(f"{len(revision['missing_dates'])} calendar days have no finalized workout. Phase labels alone are not prescriptions.")
        for change in revision['changes']:
            with st.container(border=True):
                st.write(f"{change.get('date','')} · {change.get('title','Training context')}")
                st.caption(change['type'].replace('_',' ').title())
                if 'before' in change: st.write(f"{change['before'] or '—'} → {change.get('after') or '—'}")
                if change.get('completed_match_preserved'): st.caption('Completed activity match preserved; prior instructions remain auditable.')
        if not revision['reviewed_at'] and st.button('Mark changes reviewed',key='review-plan-revision'):
            review_revision(revision['id']); st.cache_data.clear();st.rerun()
        with connect() as connection:
            old=connection.execute('SELECT id,imported_at,status,session_count FROM plan_revisions ORDER BY id DESC LIMIT 10').fetchall()
        for row in old: st.caption(f"Revision {row['id']} · {row['imported_at'][:10]} · {row['status']} · {row['session_count']} sessions")


def session_controls(session_id, *, key_prefix='session'):
    with connect() as connection:
        session=connection.execute('SELECT * FROM planned_sessions WHERE id=?',(session_id,)).fetchone()
        if not session: return
        session=dict(session)
        session['targets']=[dict(r) for r in connection.execute('SELECT * FROM planned_session_targets WHERE planned_session_id=?',(session_id,))]
        p=connection.execute("SELECT * FROM workout_prescriptions WHERE planned_session_id=? AND status='active'",(session_id,)).fetchone()
        completed=connection.execute('SELECT 1 FROM planned_activity_matches WHERE planned_session_id=? AND is_selected=1',(session_id,)).fetchone()
    completed=completed or (str(session.get('source_status') or '').lower() in {'completed','modified'} and bool(json.loads(session.get('workbook_actual_json') or '{}').get('Actual Session')))
    layers=resolve_session(session,dict(p) if p else None,training_decisions().get(session_id),lifecycle_rows().get(session_id))
    prefix=f'{key_prefix}-{session_id}'
    with st.expander('Workout source & athlete controls',expanded=layers['needs_review']):
        st.caption(f"Effective source: {layers['effective_source']} · {layers['lifecycle'].upper()}")
        st.write('Effective:',_label(layers['effective']))
        st.caption('Coach: '+_label(layers['coach_plan']))
        st.caption('Endurance Lab: '+_label(layers['app_recommendation']))
        if layers['athlete_decision']:
            st.caption('Me: '+layers['athlete_decision']['action'].replace('_',' ').title()+' · '+layers['athlete_decision']['reason'])
        if layers['needs_review']:
            st.warning('The coach materially revised the instructions behind your choice. Your choice remains in effect until you review it.')
            a,b=st.columns(2)
            if a.button('Keep my decision',key=prefix+'-retain',width='stretch'):
                d=layers['athlete_decision']
                choose_training(session_id,'RETAIN',note='Reviewed coach revision; keep my decision')
                _refresh();st.rerun()
            if b.button('Use new coach plan' if session['active'] else 'Follow coach removal',key=prefix+'-new',width='stretch'):
                choose_training(session_id,'KEEP_ORIGINAL' if session['active'] else 'CLEAR',note='Reviewed coach revision; follow new coach plan')
                _refresh();st.rerun()
        if not completed and session['active']:
            if layers['lifecycle']=='draft' and st.button('Confirm coach session',key=prefix+'-confirm'):
                confirm_session(session_id);_refresh();st.rerun()
            actions=['Accept app recommendation','Use original coach plan','Modify','Move','Skip','Clear decision']
            if layers['athlete_decision']: actions.insert(0,'Keep my decision')
            selected=st.selectbox('Your action',actions,key=prefix+'-action')
            base=layers['effective'] or layers['coach_plan']
            with st.form(prefix+'-form'):
                workout=dict(base)
                replace_decision=False
                if selected=='Accept app recommendation':
                    workout=layers['app_recommendation']
                    st.caption('App recommendation to accept: '+_label(workout))
                    if layers['athlete_decision']:
                        st.warning('This replaces your saved choice, not confirms the effective session shown above.')
                        replace_decision=st.checkbox('Replace my saved decision with this app recommendation',key=prefix+'-replace')
                if selected in {'Modify','Move'}:
                    value=st.date_input('Workout date',value=date.fromisoformat(base['date']))
                    workout['date']=value.isoformat()
                if selected=='Modify':
                    sports=['cycling','running','swimming','strength','other']
                    workout['sport']=st.selectbox('Sport',sports,index=sports.index(base.get('sport','other')))
                    workout['title']=st.text_input('Workout title',value=base.get('title') or '')
                    workout['duration_seconds']=st.number_input('Duration (minutes)',min_value=0,value=int((base.get('duration_seconds') or 0)/60))*60
                    workout['intensity']=st.text_input('Intensity',value=base.get('intensity') or '')
                    workout['description']=st.text_area('Session description',value=base.get('description') or '')
                    # Old targets must not masquerade as targets for a modified workout.
                    workout['targets']={};workout['interval_structure']=None
                note=st.text_input('Optional reason / note')
                submitted=st.form_submit_button('Save decision',width='stretch')
            if submitted:
                action={'Keep my decision':'RETAIN','Accept app recommendation':'ACCEPT','Use original coach plan':'KEEP_ORIGINAL','Modify':'MODIFY','Move':'MOVE','Skip':'SKIP','Clear decision':'CLEAR'}[selected]
                try:
                    choose_training(session_id,action,workout=workout,note=note or None,replace_decision=replace_decision)
                    _refresh();st.rerun()
                except ValueError as exc: st.error(str(exc))
        else:
            st.caption('Completed sessions are read-only; their activity matches are preserved.')
        with st.expander('Session history'):
            history=session_history(session_id)
            for version in history['coach_revisions']:
                snapshot=json.loads(version['snapshot_json'])
                st.caption(f"Coach revision {version['revision_id']} · {version['imported_at'][:10]} · {snapshot['title']} · {format_duration(snapshot.get('planned_duration_seconds'))}")
            for item in history['app_prescriptions']:
                st.caption(f"Endurance Lab · {item['created_at'][:16]} · {item['action']} · {item['status']}")
            for item in history['athlete_decisions']:
                st.caption(f"Me · {item['created_at'][:16]} · {item['action']} · {item.get('note') or ''}")
