from datetime import date, timedelta
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard import plan_pages


def test_plan_week_navigation_updates_summary_and_respects_plan_bounds(monkeypatch):
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    first = monday - timedelta(days=14)
    last = monday + timedelta(days=14)
    calls = []
    monkeypatch.setattr(plan_pages, "_status", lambda: {
        "plan": {"date_start": first.isoformat(), "date_end": (last + timedelta(days=6)).isoformat()}
    })
    monkeypatch.setattr(plan_pages, "_sessions", lambda start, end: [])

    def weeks(start, end):
        calls.append((start, end))
        previous = start < monday
        return [{
            "week_start": start.isoformat(), "phase": "Test phase", "week_type": "Build",
            "completed_sessions": 1 if previous else 0, "planned_sessions": 7,
            "planned_duration_seconds": 7200, "actual_duration_seconds": 3600 if previous else 0,
            "key_sessions_completed": 0, "key_sessions_planned": 2,
            "strength_sessions_completed": 0, "strength_sessions_planned": 1,
            "planned_load": None, "actual_load": 10,
            "planned_distance_by_sport": {}, "actual_distance_by_sport": {},
        }]

    monkeypatch.setattr(plan_pages, "_weeks", weeks)
    app = AppTest.from_file(str(Path(__file__).parent / "fixtures" / "plan_page_app.py"))
    app.run()
    assert not app.exception
    assert app.session_state["plan_week_start"] == monday
    assert app.metric[1].value == "0m"

    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["plan_week_start"] == monday - timedelta(days=7)
    assert app.metric[1].value == "1h 00m"
    assert calls[-1] == (monday - timedelta(days=7), monday - timedelta(days=1))

    app.button[0].click().run()
    assert app.button[0].disabled
    app.button[1].click().run()
    assert app.session_state["plan_week_start"] == monday
    app.button[2].click().run()
    app.button[2].click().run()
    assert app.session_state["plan_week_start"] == last
    assert app.button[2].disabled
    assert not app.exception
