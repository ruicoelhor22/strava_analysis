import json
from datetime import date

import pytest

from endurance_lab.app_planning import save_app_plan
from endurance_lab.db import connect
from endurance_lab.plan_adherence import planned_sessions, plan_status
from endurance_lab.plan_revisions import sync_plan_revision
from tests.test_plan_lifecycle import fixture


def proposal(root, sport='running'):
    source=root/'app-plan.json'
    source.write_text(json.dumps({'name':'App travel maintenance','date_start':'2026-10-19','date_end':'2026-10-19',
        'rationale':'Explicitly requested app planning, not a coach workbook.',
        'restrictions':[{'start':'2026-10-18','end':'2026-10-23','allowed_sports':['running','strength']}],
        'sessions':[{'date':'2026-10-19','sport':sport,'title':'Easy run','minutes':25,'intensity':'Easy','priority':'C - Optional','description':'Easy flat run-walk.'}]}),encoding='utf-8')
    return source


def test_supplement_is_idempotent_and_does_not_replace_coach_source():
    root,workbook,database=fixture()
    source=proposal(root)
    with connect(database) as c:
        original=[dict(r) for r in c.execute('SELECT * FROM planned_sessions')]
    ids=save_app_plan(source,database)
    assert ids==save_app_plan(source,database)
    assert sync_plan_revision(database)['status']=='unchanged'
    with connect(database) as c:
        assert original==[dict(r) for r in c.execute('SELECT * FROM planned_sessions WHERE source_sheet!=?',('App planning',))]
    rows=planned_sessions(database,start=date(2026,10,19),end=date(2026,10,19))
    assert len(rows)==1 and rows[0]['effective_source']=='APP PLAN'
    status=plan_status(database)
    assert status['plan']['date_start']<='2026-10-19'<=status['plan']['date_end']


def test_travel_rejects_bike_before_database_mutation():
    root,_,database=fixture()
    with connect(database) as c:
        count=c.execute('SELECT COUNT(*) FROM training_plans').fetchone()[0]
    with pytest.raises(ValueError,match='travel availability'):
        save_app_plan(proposal(root,'cycling'),database)
    with connect(database) as c:
        assert c.execute('SELECT COUNT(*) FROM training_plans').fetchone()[0]==count


def test_rest_after_key_session_is_not_conditional_training():
    from endurance_lab.weekly_adjustment import adjust_week
    from tests.test_coaching import _database, _plan_session
    database=_database()
    _plan_session(database,'2026-10-17','cycling','Long endurance ride',9000,priority='A - Key')
    rest=_plan_session(database,'2026-10-18','other','Travel day - rest',0,priority='C - Optional')
    item=next(p for p in adjust_week(date(2026,10,12),database,as_of=date(2026,10,12),persist=False) if p.planned_session_id==rest)
    assert item.action=='KEEP' and item.optional_gate is None
    assert item.prescribed['duration_seconds']==0
