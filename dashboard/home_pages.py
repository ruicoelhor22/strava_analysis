from __future__ import annotations

from datetime import date, timedelta
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import (
    activity_card, cost_indicator, format_duration, metric_tile, page_intro,
    section_header, sport_marker, status_badge, style_figure,
)
from endurance_lab.athlete_state import athlete_state
from endurance_lab.db import connect
from endurance_lab.plan_adherence import plan_status, planned_sessions
from endurance_lab.prescription import prescribe_day
from endurance_lab.queries import daily_load


@st.cache_data(ttl=60, show_spinner=False)
def _home_context(today: date) -> tuple[dict, list[dict], list[dict], dict, dict[int, dict]]:
    monday = today - timedelta(days=today.weekday())
    sessions = planned_sessions(start=monday, end=monday + timedelta(days=6), today=today)
    ids = [int(item["id"]) for item in sessions]
    prescribed: dict[int, dict] = {}
    if ids:
        placeholders = ",".join("?" for _ in ids)
        with connect() as connection:
            prescribed = {
                int(row["planned_session_id"]): dict(row)
                for row in connection.execute(
                    f"SELECT planned_session_id, title, action, duration_seconds "
                    f"FROM workout_prescriptions WHERE status='active' "
                    f"AND planned_session_id IN ({placeholders})", ids
                )
            }
    return (
        athlete_state(today).to_dict(),
        [item.to_dict() for item in prescribe_day(today, persist=False)],
        sessions,
        plan_status(today=today),
        prescribed,
    )


@st.cache_data(ttl=60, show_spinner=False)
def _recent_costs(activity_ids: tuple[int, ...]) -> dict[int, dict]:
    if not activity_ids:
        return {}
    placeholders = ",".join("?" for _ in activity_ids)
    with connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM session_costs WHERE activity_id IN ({placeholders})", activity_ids
        ).fetchall()
    return {int(row["activity_id"]): dict(row) for row in rows}


def _go(page: str, activity_id: int | None = None) -> None:
    st.session_state["primary_nav"] = page
    st.session_state["system_view"] = False
    if activity_id is not None:
        st.session_state["selected_activity_id"] = activity_id


def home_page(frame: pd.DataFrame) -> None:
    today = date.today()
    state, prescriptions, week, status, prescribed = _home_context(today)
    recent = frame.head(8)
    costs = _recent_costs(tuple(int(value) for value in recent["id"])) if not recent.empty else {}
    page_intro("Performance lab / live", "Today", today.strftime("%A · %d %B %Y"))

    if prescriptions:
        primary = prescriptions[0]
        workout = primary.get("prescribed") or primary.get("original") or {}
        reasons = primary.get("reasons") or []
        duration = format_duration(workout.get("duration_seconds"))
        targets = workout.get("targets") or {}
        target_text = " · ".join(str(value) for value in targets.values() if value)
        title = workout.get("title") or "Session"
        action = str(primary.get("action") or "KEEP")
        tone = "good" if action == "KEEP" else "watch"
        st.markdown(
            '<div class="lab-hero">'
            f'<div class="lab-kicker">TODAY · {escape(sport_marker(workout.get("sport", "other")))}</div>'
            f'<div class="lab-hero-title">{escape(str(title))}</div>'
            f'<div class="lab-facts"><span>{escape(duration)}</span>'
            f'<span>{escape(str(workout.get("intensity") or "Intensity unspecified"))}</span>'
            f'<span>{status_badge(action, tone)}</span></div>'
            f'<div class="lab-sub">{escape(str(reasons[0])) if reasons else "Plan retained."}</div>'
            f'<div class="lab-sub">{escape(target_text)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.button("View today's decision", on_click=_go, args=("Coach",), key="home-coach")
    else:
        st.markdown(
            '<div class="lab-hero"><div class="lab-kicker">TODAY</div>'
            '<div class="lab-hero-title">No prescribed session</div>'
            '<div class="lab-sub">The coaching plan does not specify a workout for today.</div></div>',
            unsafe_allow_html=True,
        )

    section_header("Current state", "Four signals from measured training and the coaching plan.")
    ratio = state["load"].get("recent_to_reference_ratio")
    load_label = "Unknown" if ratio is None else "Elevated" if ratio >= 1.3 else "Low" if ratio < .75 else "Normal"
    load_tone = "neutral" if ratio is None else "watch" if ratio >= 1.3 else "good"
    legs_hours = state["recovery_context"].get("hours_since_leg_strength")
    leg_label = "Unknown" if legs_hours is None else "Loaded" if legs_hours < 36 else "Recovering" if legs_hours < 72 else "No recent load"
    leg_tone = "watch" if legs_hours is not None and legs_hours < 36 else "neutral"
    cycling_trend = (state["performance"].get("cycling") or {})
    trend = str(cycling_trend.get("state") or "UNCERTAIN").title()
    week_data = status.get("current_week") or {}
    completed = week_data.get("completed_sessions")
    planned = week_data.get("planned_sessions")
    plan_text = f"{completed} / {planned}" if planned else "Unknown"
    st.markdown(
        '<div class="lab-state-grid">'
        + metric_tile("Load", load_label, f"{ratio:.2f}× recent baseline" if ratio is not None else "Baseline unavailable", load_tone)
        + metric_tile("Legs", leg_label, f"{legs_hours:.0f} h since lower-body strength" if legs_hours is not None else "No recent strength evidence", leg_tone)
        + metric_tile("Cycling trend", trend, str(cycling_trend.get("confidence") or "unknown") + " confidence")
        + metric_tile("Plan", plan_text, "sessions linked this week")
        + '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Load and trend evidence"):
        load = state["load"]
        st.write(
            f"7-day estimated load: {load.get('load_7d', 0):.0f}. "
            f"Reference week: {load.get('weekly_reference_load'):.0f}."
            if load.get("weekly_reference_load") else
            f"7-day estimated load: {load.get('load_7d', 0):.0f}; reference unavailable."
        )
        for signal in state["performance"].values():
            st.caption(f"{signal['sport'].title()}: {'; '.join(signal.get('evidence') or [])}")

    section_header("This week", "Planned, prescribed and completed work in one scan.")
    monday = today - timedelta(days=today.weekday())
    by_day: dict[str, list[dict]] = {}
    for item in week:
        by_day.setdefault(str(item["planned_date"]), []).append(item)
    week_html = []
    for offset in range(7):
        day = monday + timedelta(days=offset)
        items = by_day.get(day.isoformat(), [])
        if not items:
            title = "No planned session"
            detail = ""
            badge = status_badge("Today" if day == today else "Open", "neutral")
        else:
            title = " · ".join(
                f"{sport_marker(item['sport'])} "
                f"{(prescribed.get(int(item['id'])) or {}).get('title') or item['title']}"
                for item in items
            )
            detail = " · ".join(
                f"{format_duration((prescribed.get(int(item['id'])) or {}).get('duration_seconds') or item.get('planned_duration_seconds'))} "
                f"{(prescribed.get(int(item['id'])) or {}).get('action') or 'KEEP'} · "
                f"{item.get('priority') or 'No priority'}"
                for item in items
            )
            complete = all(
                item["completion_state"] in {"completed", "completed_metadata", "partial"}
                for item in items
            )
            key = any(str(item.get("priority") or "").startswith("A") for item in items)
            badge = status_badge(
                "Done" if complete else "Today" if day == today else "Key" if key else "Planned",
                "good" if complete else "watch" if key or day == today else "neutral",
            )
        week_html.append(
            f'<div class="lab-week"><div class="lab-week-day">{day.strftime("%a")}<br>{day.day:02d}</div>'
            f'<div class="lab-week-main"><strong>{escape(title)}</strong><small>{escape(detail)}</small></div>'
            f'{badge}</div>'
        )
    st.markdown('<div class="lab-row">' + "".join(week_html) + '</div>', unsafe_allow_html=True)
    st.button("Open training plan", on_click=_go, args=("Plan",), key="home-plan")

    if not frame.empty:
        latest = frame.iloc[0].to_dict()
        latest_cost = costs.get(int(latest["id"]))
        section_header("Latest activity", "The newest imported training session.")
        st.markdown(activity_card(latest, latest_cost), unsafe_allow_html=True)
        st.button(
            "Open activity", on_click=_go, args=("Activities", int(latest["id"])),
            key="home-activity",
        )
        if latest_cost:
            with st.expander("Training cost dimensions"):
                st.markdown(cost_indicator(latest_cost), unsafe_allow_html=True)

    section_header("Coach insight", "The strongest reason behind today's decision.")
    reason = (prescriptions[0].get("reasons") or [None])[0] if prescriptions else None
    st.markdown(
        '<div class="lab-insight">'
        + escape(str(reason or "No active rule changed today's plan."))
        + '</div>',
        unsafe_allow_html=True,
    )
    if prescriptions:
        with st.expander("Why?"):
            for item in prescriptions:
                for line in item.get("reasons") or []:
                    st.write(line)
                for protected in item.get("protected_sessions") or []:
                    st.caption(f"Protected: {protected.get('date')} · {protected.get('title')}")

    section_header("Training load · last 6 weeks", "Is recent load rising faster than its baseline?")
    load_frame = daily_load()
    if load_frame.empty:
        st.caption("Load trend unavailable.")
    else:
        chart = load_frame[load_frame["day"] >= pd.Timestamp(today - timedelta(days=42))]
        fig = go.Figure()
        fig.add_bar(x=chart["day"], y=chart["total_load"], marker_color="#40556a", name="Daily")
        fig.add_scatter(x=chart["day"], y=chart["fitness"], line_color="#73e0a9", name="Fitness")
        fig.update_layout(showlegend=False, yaxis_title="Estimated load")
        st.plotly_chart(style_figure(fig, 245), width="stretch", config={"displayModeBar": False})
