from __future__ import annotations

from datetime import date, timedelta
from html import escape

import pandas as pd
import streamlit as st

from dashboard._shared import cost_indicator, decision_pill, format_duration, page_intro, section_header, sport_marker
from endurance_lab.athlete_state import athlete_state
from endurance_lab.plan_adherence import planned_sessions
from endurance_lab.prescription import prescribe_day
from endurance_lab.subjective import get_checkin, save_checkin
from endurance_lab.weekly_adjustment import adjust_week


@st.cache_data(ttl=30, show_spinner=False)
def _state(value: date) -> dict:
    return athlete_state(value).to_dict()


@st.cache_data(ttl=30, show_spinner=False)
def _day(value: date) -> list[dict]:
    return [item.to_dict() for item in prescribe_day(value, persist=False)]


@st.cache_data(ttl=30, show_spinner=False)
def _week(value: date, as_of: date) -> list[dict]:
    return [item.to_dict() for item in adjust_week(value, as_of=as_of, persist=False)]


def coach_page() -> None:
    today = date.today()
    page_intro("Endurance coach / decision trace", "Coach", "What matters now, today's decision, and the evidence behind it.")
    prescriptions = _day(today)
    state = _state(today)

    section_header("What matters now")
    reasons = [reason for item in prescriptions for reason in item.get("reasons") or []]
    st.markdown(
        '<div class="lab-insight">' + escape(str(reasons[0] if reasons else
        "The original plan remains suitable; no adjustment rule is active.")) + '</div>',
        unsafe_allow_html=True,
    )
    section_header("Today's prescription", "The imported plan, adjusted only when a transparent coaching rule is triggered.")
    if not prescriptions:
        st.info("There is no explicit workout in the imported coaching plan today. The engine does not invent one.")
    for item in prescriptions:
        _prescription_card(item)

    section_header("Why", "The evidence available before today's workout.")
    left, right = st.columns([1.35, 1])
    with left:
        with st.container(border=True):
            st.subheader("Why this decision")
            if not reasons:
                st.caption("The original plan is retained; no adaptation rule is active.")
            for reason in reasons[:5]:
                st.markdown(f"**→** {reason}")
    with right:
        with st.container(border=True):
            st.subheader("Decision context")
            for name, dimension in state["dimensions"].items():
                label = name.replace("_", " ").title()
                tone = "attention" if dimension["state"] == "concern" else "neutral"
                st.markdown(
                    f"{decision_pill(dimension['state'], tone)} &nbsp; **{label}**",
                    unsafe_allow_html=True,
                )
                if dimension["reasons"]:
                    st.caption(dimension["reasons"][0])

    with st.expander("Athlete state · full context"):
        load = state["load"]
        adherence = state["adherence"]
        recovery = state["recovery_context"]
        load_ratio = load.get("recent_to_reference_ratio")
        completion_ratio = adherence.get("completion_ratio_28d")
        st.write(
            f"7-day load: {load.get('load_7d', 0):.0f} · "
            f"Reference ratio: {load_ratio:.2f}×" if load_ratio is not None
            else f"7-day load: {load.get('load_7d', 0):.0f} · Reference unavailable"
        )
        st.write(
            f"28-day plan completion: {completion_ratio:.0%}" if completion_ratio is not None
            else "28-day plan completion unavailable"
        )
        st.caption(
            f"Last training: {recovery.get('last_activity_date') or 'unavailable'} · "
            f"Data confidence: {state['data_quality']['confidence']}"
        )

    section_header("Schedule", "A single view of the original plan, current prescription, and completed work.")
    upcoming = _adaptation_rows(today, today + timedelta(days=6))
    monday = today - timedelta(days=today.weekday())
    week = _adaptation_rows(monday, monday + timedelta(days=6))
    if upcoming.empty:
        st.caption("No explicit planned sessions in the next seven days.")
    else:
        for row in upcoming.head(7).to_dict("records"):
            st.markdown(
                '<div class="lab-week"><div class="lab-week-day">'
                + escape(pd.Timestamp(row["date"]).strftime("%a %d"))
                + '</div><div class="lab-week-main"><strong>'
                + escape(f"{sport_marker(row['sport'].lower())} · {row['current prescription']}")
                + '</strong><small>' + escape(f"{row['decision']} · {row['why']}")
                + '</small></div></div>', unsafe_allow_html=True,
            )
        with st.expander("Full schedule and current week"):
            st.dataframe(upcoming, width="stretch", hide_index=True)
            if not week.empty:
                st.dataframe(week, width="stretch", hide_index=True)

    _cost_and_reconciliation(today)

    _checkin_form(today)


def _prescription_card(item: dict) -> None:
    workout = item["prescribed"]
    action = item["action"].replace("_", " ").title()
    tone = "good" if item["action"] == "KEEP" else "watch"
    original_duration = item["original"].get("duration_seconds")
    prescribed_duration = workout.get("duration_seconds")
    with st.container(border=True):
        heading, decision = st.columns([4, 1])
        with heading:
            st.markdown(
                f'<div class="workout-kicker">{escape(str(workout["sport"]).title())} · '
                f'{escape(str(workout.get("intensity") or "Planned workout"))}</div>'
                f'<div class="workout-title">{escape(str(workout["title"]))}</div>'
                f'<div class="workout-meta">{format_duration(prescribed_duration)} · '
                f'{escape(str(workout.get("priority") or "No priority"))}</div>',
                unsafe_allow_html=True,
            )
        with decision:
            st.markdown(decision_pill(action, tone), unsafe_allow_html=True)
        if original_duration and prescribed_duration and original_duration != prescribed_duration:
            st.caption(
                f"Adjusted from {format_duration(original_duration)} to {format_duration(prescribed_duration)}; "
                "the main work remains the priority."
            )
        if workout.get("description"):
            st.write(workout["description"])
        if workout.get("targets"):
            target_columns = st.columns(min(3, len(workout["targets"])))
            for column, (target_type, target) in zip(target_columns, workout["targets"].items()):
                column.markdown(f"**{target_type.replace('_', ' ').title()}**")
                column.caption(target)
        with st.expander("Workout structure and decision details"):
            if workout.get("interval_structure"):
                st.write(workout["interval_structure"])
            st.caption(f"Confidence: {item['confidence']} · Rules: {', '.join(item['rules_triggered']) or 'none'}")


def _schedule_cards(frame: pd.DataFrame) -> None:
    columns = st.columns(min(3, len(frame)))
    for column, (_, row) in zip(columns, frame.iterrows()):
        with column.container(border=True):
            day = pd.Timestamp(row["date"]).strftime("%a · %d %b")
            st.caption(day.upper())
            st.markdown(f"**{row['current prescription']}**")
            st.caption(f"{row['sport']} · {row['decision']}")
            st.caption(str(row["why"]))


def _adaptation_rows(start: date, end: date) -> pd.DataFrame:
    sessions = planned_sessions(start=start, end=end)
    mondays = {
        value - timedelta(days=value.weekday())
        for value in (start, end)
    }
    prescription_map = {
        item["planned_session_id"]: item
        for monday in mondays
        for item in _week(monday, date.today())
        if start.isoformat() <= item["date"] <= end.isoformat()
    }
    rows = []
    for session in sessions:
        item = prescription_map.get(session["id"])
        actual = session.get("actual_name") or session.get("workbook_actual", {}).get("Actual Session")
        prescribed = (item or {}).get("prescribed") or {}
        rows.append({
            "date": session["planned_date"],
            "sport": str(session["sport"]).title(),
            "original plan": session["title"],
            "current prescription": prescribed.get("title") or ("No replacement — not training debt" if item and item["action"] == "SKIP" else session["title"]),
            "decision": item["action"].replace("_", " ").title() if item else "Keep",
            "status": session["completion_state"].replace("_", " ").title(),
            "completed": actual or session["completion_state"].replace("_", " ").title(),
            "why": item["reasons"][0] if item and item["reasons"] else "Plan retained",
        })
    return pd.DataFrame(rows)


def _cost_and_reconciliation(today: date) -> None:
    monday = today - timedelta(days=today.weekday())
    decisions = _week(monday, today)
    section_header(
        "Cost-aware week adjustment",
        "Actual training cost is separated into cardiovascular, muscular, sport-specific, intensity, and duration dimensions.",
    )
    actual_by_id: dict[int, dict] = {}
    for item in decisions:
        for actual in (item.get("decision_trace") or {}).get("actual_context", []):
            if actual["date"] >= monday.isoformat():
                actual_by_id[int(actual["activity_id"])] = actual
    if actual_by_id:
        cost_rows = []
        for index, actual in enumerate(sorted(actual_by_id.values(), key=lambda value: value["date"], reverse=True)):
            cost = actual["session_cost"]
            if index < 4:
                st.markdown(
                    '<div class="lab-row"><div class="lab-row-head">'
                    + escape(f"{actual['date']} · {actual['sport']}")
                    + '</div><div class="lab-row-title">'
                    + escape(str(actual.get("name") or actual["activity_id"]))
                    + '</div>' + cost_indicator(cost) + '</div>',
                    unsafe_allow_html=True,
                )
            cost_rows.append({
                "date": actual["date"], "activity": str(actual.get("name") or actual["activity_id"]),
                "sport": str(actual["sport"]).title(), "overall cost": cost["cost_class"].title(),
                "systemic": cost["systemic_cost"].title(), "cardiovascular": cost["cardiovascular_cost"].title(),
                "muscular": f"{cost['muscular_cost'].title()} · {cost['muscle_load']}",
                "intensity": cost["intensity_cost"].title(), "duration": cost["duration_cost"].title(),
                "confidence": cost["confidence"].title(),
            })
        with st.expander("Full cost table"):
            st.dataframe(pd.DataFrame(cost_rows), width="stretch", hide_index=True)
    else:
        st.caption("No completed activities are available before today's evidence cut-off.")

    for item in decisions:
        original = item.get("original") or {}
        prescribed = item.get("prescribed") or {}
        actual = (item.get("decision_trace") or {}).get("actual_context", [])
        label = f"{item['date']} · {original.get('title', 'No plan')} → {item['action'].replace('_', ' ').title()}"
        with st.expander(label):
            columns = st.columns(3)
            columns[0].markdown("**Original plan**")
            columns[0].write(original.get("title") or "No explicit plan")
            columns[1].markdown("**Current prescription**")
            columns[1].write(prescribed.get("title") or "No session prescribed")
            columns[2].markdown("**Actual evidence available**")
            columns[2].write(", ".join(str(value.get("name") or value["activity_id"]) for value in actual) or "None")
            for reason in item.get("reasons", []):
                st.markdown(f"- {reason}")
            trace = item.get("decision_trace") or {}
            st.caption(
                f"Priority: {(trace.get('planned_session') or {}).get('priority', '—')} "
                f"({(trace.get('planned_session') or {}).get('priority_source', 'inferred')}) · "
                f"Recovery runway: {item.get('recovery_runway_h') or '—'} h · "
                f"Confidence: {item.get('confidence', '—')}"
            )
            if item.get("optional_gate"):
                gate = item["optional_gate"]
                st.info(f"Optional gate: {gate['state']}. Fallback: {gate['fallback']['title']}.")


def _checkin_form(today: date) -> None:
    with st.expander("Optional daily check-in"):
        existing = get_checkin(today)
        if existing:
            st.caption("Today's check-in is saved locally. Submitting again updates it.")
        choices = ["Not provided", 1, 2, 3, 4, 5]
        with st.form("daily-checkin"):
            columns = st.columns(3)
            sleep = columns[0].selectbox("Sleep quality", choices, index=_choice_index(existing, "sleep_quality"))
            fatigue = columns[1].selectbox("Fatigue", choices, index=_choice_index(existing, "fatigue"))
            soreness = columns[2].selectbox("Leg soreness", choices, index=_choice_index(existing, "leg_soreness"))
            stress = columns[0].selectbox("Stress", choices, index=_choice_index(existing, "stress"))
            motivation = columns[1].selectbox("Motivation", choices, index=_choice_index(existing, "motivation"))
            recovery = columns[2].selectbox("Recovery RPE", choices, index=_choice_index(existing, "recovery_rpe"))
            notes = st.text_area("Notes", value=str((existing or {}).get("notes") or ""))
            submitted = st.form_submit_button("Save check-in")
        if submitted:
            save_checkin(
                today,
                sleep_quality=_optional_score(sleep), fatigue=_optional_score(fatigue),
                leg_soreness=_optional_score(soreness), stress=_optional_score(stress),
                motivation=_optional_score(motivation), recovery_rpe=_optional_score(recovery),
                notes=notes,
            )
            st.cache_data.clear()
            st.success("Check-in saved locally.")
            st.rerun()


def _choice_index(existing: dict | None, key: str) -> int:
    value = (existing or {}).get(key)
    return int(value) if value in {1, 2, 3, 4, 5} else 0


def _optional_score(value) -> int | None:
    return None if value == "Not provided" else int(value)
