from __future__ import annotations

from datetime import date, timedelta
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import COLORS, decision_pill, format_duration, page_intro, section_header, sport_marker, style_figure
from endurance_lab.coaching import coaching_flags
from endurance_lab.db import connect
from endurance_lab.plan_adherence import plan_status, planned_sessions, weekly_adherence
from endurance_lab.queries import activities, daily_load, power_history


@st.cache_data(ttl=30, show_spinner=False)
def _sessions(start: date, end: date) -> list[dict]:
    return planned_sessions(start=start, end=end)


@st.cache_data(ttl=30, show_spinner=False)
def _weeks(start: date, end: date) -> list[dict]:
    return weekly_adherence(start=start, end=end)


@st.cache_data(ttl=30, show_spinner=False)
def _status() -> dict:
    return plan_status()


@st.cache_data(ttl=30, show_spinner=False)
def _flags() -> list[dict]:
    return coaching_flags()


@st.cache_data(ttl=30, show_spinner=False)
def _actual_activities(start: date, end: date) -> pd.DataFrame:
    return activities(start=start, end=end + timedelta(days=1))


def training_plan_page() -> None:
    page_intro(
        "Planned · prescribed · completed",
        "Training plan",
        "The coaching calendar linked conservatively to measured activities and metadata-only completions.",
    )
    status = _status()
    if not status.get("plan"):
        st.info("No coaching plan has been imported yet.")
        return
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    current = _sessions(monday, sunday)
    week = status.get("current_week")

    if week:
        with st.container(border=True):
            top, badge = st.columns([4, 1])
            top.markdown(f"**{week.get('phase') or 'Current block'}** · {week.get('week_type') or 'Unspecified week'}")
            badge.markdown(decision_pill("Current week", "good"), unsafe_allow_html=True)
            completion = week["completed_sessions"] / week["planned_sessions"] if week["planned_sessions"] else 0
            st.progress(completion)
            st.caption(f"{week['completed_sessions']} of {week['planned_sessions']} sessions linked · {completion:.0%} complete")

        cols = st.columns(4)
        cols[0].metric("Planned time", format_duration(week["planned_duration_seconds"]))
        cols[1].metric(
            "Completed time",
            format_duration(week["actual_duration_seconds"]),
            f"{week['actual_duration_seconds'] / week['planned_duration_seconds']:.0%} of plan" if week["planned_duration_seconds"] else None,
            delta_color="off",
        )
        cols[2].metric(
            "Key sessions",
            f"{week['key_sessions_completed']} / {week['key_sessions_planned']}",
        )
        cols[3].metric(
            "Strength",
            f"{week['strength_sessions_completed']} / {week['strength_sessions_planned']}",
        )
        planned_load = week.get("planned_load")
        load_text = (
            f"{week['actual_load']:.0f} actual / {planned_load:.0f} planned"
            if planned_load is not None else f"{week['actual_load']:.0f} actual / planned unavailable"
        )
        st.caption(
            f"Distance - planned: {_distance_summary(week['planned_distance_by_sport'])}; "
            f"actual: {_distance_summary(week['actual_distance_by_sport'])}. "
            f"Load - {load_text}. Strength: {week['strength_sessions_completed']} / "
            f"{week['strength_sessions_planned']} completed."
        )

    section_header("Current week", "Original plan, current prescription and completion at a glance.")
    if not current:
        st.caption("No explicit sessions are prescribed for this week.")
    else:
        _week_grid(current, monday, today)
        with st.expander("Full week table"):
            st.dataframe(_session_table(current), width="stretch", hide_index=True)

    section_header("Plan timeline", "Scan completion state by sport, then inspect the underlying schedule.")
    weeks_back = st.radio(
        "Window", ["4 weeks", "12 weeks", "Full plan"], index=1, horizontal=True
    )
    if weeks_back == "4 weeks":
        start, end = today - timedelta(days=14), today + timedelta(days=14)
    elif weeks_back == "Full plan":
        start = date.fromisoformat(status["plan"]["date_start"])
        end = date.fromisoformat(status["plan"]["date_end"])
    else:
        start, end = today - timedelta(days=28), today + timedelta(days=56)
    timeline = _sessions(start, end)
    if timeline:
        frame = pd.DataFrame(timeline)
        frame["planned_date"] = pd.to_datetime(frame["planned_date"])
        frame["timeline_state"] = frame.apply(
            lambda row: "ambiguous" if row.get("match_status") == "ambiguous" else row["completion_state"],
            axis=1,
        )
        state_colors = {
            "completed": COLORS["cycling"], "completed_metadata": "#8FD3FF",
            "partial": "#F7C873", "missed": COLORS["fatigue"],
            "planned": "#EAF0F5", "upcoming": "#8190A5", "ambiguous": "#F59E0B",
            "unknown": "#64748B",
        }
        fig = go.Figure()
        for state, group in frame.groupby("timeline_state"):
            fig.add_scatter(
                x=group["planned_date"], y=group["sport"].str.title(), mode="markers",
                name=state.replace("_", " ").title(),
                marker=dict(size=12, color=state_colors.get(state, "#8190A5"), symbol="square"),
                customdata=group[["title", "priority"]],
                hovertemplate="%{x|%d %b}<br>%{y}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            )
        actual_frame = _actual_activities(start, end)
        matched_ids = {int(item["activity_id"]) for item in timeline if item.get("activity_id")}
        if not actual_frame.empty:
            extras = actual_frame[~actual_frame["id"].isin(matched_ids)]
            if not extras.empty:
                fig.add_scatter(
                    x=extras["started_at"], y=extras["sport"].str.title(), mode="markers",
                    name="Extra / unmatched actual",
                    marker=dict(size=11, color="#A78BFA", symbol="diamond-open", line_width=2),
                    customdata=extras[["name"]],
                    hovertemplate="%{x|%d %b}<br>%{y}<br>%{customdata[0]}<extra></extra>",
                )
        st.plotly_chart(style_figure(fig, 300), width="stretch")
        with st.expander("Full plan table"):
            st.dataframe(_session_table(timeline), width="stretch", hide_index=True)
    else:
        st.caption("No explicit planned sessions in this window.")

    section_header("Session detail", "Compare the written prescription with the linked completed activity.")
    all_sessions = _sessions(
        date.fromisoformat(status["plan"]["date_start"]),
        date.fromisoformat(status["plan"]["date_end"]),
    )
    labels = {
        f"{item['planned_date']} · {item['sport'].title()} · {item['title']}": item
        for item in all_sessions
    }
    if labels:
        label_list = list(labels)
        upcoming_index = next(
            (index for index, item in enumerate(all_sessions) if item["planned_date"] >= today.isoformat()),
            max(0, len(label_list) - 1),
        )
        selected = labels[st.selectbox("Planned session", label_list, index=upcoming_index)]
        _session_detail(selected)


def coaching_overview_page() -> None:
    page_intro(
        "Coaching context",
        "What is going on?",
        "Recent adherence, plan context, measured load, and a small set of explainable signals.",
    )
    status = _status()
    if not status.get("plan"):
        st.info("Import a coaching plan to enable planned-versus-actual context.")
        return
    today = date.today()
    recent_weeks = _weeks(today - timedelta(days=28), today)
    sessions = _sessions(today - timedelta(days=7), today + timedelta(days=21))
    current = next((item for item in sessions if item["planned_date"] >= today.isoformat()), None)
    phase = current.get("phase_name") if current else (sessions[-1].get("phase_name") if sessions else "—")
    week_type = current.get("week_type") if current else "—"
    cols = st.columns(4)
    cols[0].metric("Current block", phase or "—")
    cols[1].metric("Week type", week_type or "—")
    if recent_weeks:
        planned = sum(item["planned_sessions"] for item in recent_weeks)
        completed = sum(item["completed_sessions"] for item in recent_weeks)
        cols[2].metric("4-week completion", f"{completed / planned:.0%}" if planned else "—")
        cols[3].metric("Unmatched activities", sum(item["unmatched_activities"] for item in recent_weeks))
    else:
        cols[2].metric("4-week completion", "—")
        cols[3].metric("Unmatched activities", "—")

    section_header("Recent adherence", "Planned time versus linked completed time across recent weeks.")
    left, right = st.columns([1.5, 1])
    with left:
        if recent_weeks:
            frame = pd.DataFrame(recent_weeks)
            frame["week_start"] = pd.to_datetime(frame["week_start"])
            fig = go.Figure()
            fig.add_bar(
                x=frame["week_start"], y=frame["planned_duration_seconds"] / 3600,
                name="Planned", marker_color="#405166",
            )
            fig.add_bar(
                x=frame["week_start"], y=frame["actual_duration_seconds"] / 3600,
                name="Actual", marker_color=COLORS["cycling"],
            )
            fig.update_layout(barmode="group", yaxis_title="Hours")
            st.plotly_chart(style_figure(fig), width="stretch")
        else:
            st.caption("No recent explicit plan data.")
    with right:
        st.subheader("Active flags")
        flags = _flags()
        if not flags:
            st.caption("No deterministic flags are active.")
        for flag in flags[:8]:
            if flag["severity"] == "attention":
                st.error(flag["message"])
            elif flag["severity"] == "watch":
                st.warning(flag["message"])
            else:
                st.info(flag["message"])

    section_header("Training load context", "Daily estimated load with longer-term fitness and short-term fatigue.")
    load = daily_load()
    if load.empty:
        st.caption("Measured training load is unavailable.")
    else:
        recent = load[load["day"] >= pd.Timestamp(today - timedelta(days=42))]
        fig = go.Figure()
        fig.add_bar(x=recent["day"], y=recent["total_load"], name="Daily load", marker_color="#405166")
        fig.add_scatter(x=recent["day"], y=recent["fitness"], name="Fitness", line_color=COLORS["fitness"])
        fig.add_scatter(x=recent["day"], y=recent["fatigue"], name="Fatigue", line_color=COLORS["fatigue"])
        st.plotly_chart(style_figure(fig), width="stretch")

    section_header("Measured performance signals", "Only signals supported by the available activity and sensor data.")
    signals = _performance_signals(today)
    signal_columns = st.columns(max(1, len(signals)))
    for column, signal in zip(signal_columns, signals):
        column.metric(signal["label"], signal["value"], signal.get("delta"))
        column.caption(signal["context"])


def _session_table(items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in items:
        actual = item.get("actual_name") or item.get("workbook_actual", {}).get("Actual Session") or "—"
        rows.append(
            {
                "date": item["planned_date"],
                "sport": str(item["sport"]).title(),
                "priority": item.get("priority") or "—",
                "workout": item["title"],
                "target": _target_summary(item),
                "planned": format_duration(item.get("planned_duration_seconds")),
                "planned distance": _format_distance(item.get("planned_distance_m")),
                "state": item["completion_state"].replace("_", " ").title(),
                "actual": actual,
                "confidence": item.get("match_status") or ("metadata" if item["completion_state"] == "completed_metadata" else "—"),
            }
        )
    return pd.DataFrame(rows)


def _week_grid(items: list[dict], monday: date, today: date) -> None:
    ids = [int(item["id"]) for item in items]
    prescriptions: dict[int, dict] = {}
    if ids:
        placeholders = ",".join("?" for _ in ids)
        with connect() as connection:
            prescriptions = {
                int(row["planned_session_id"]): dict(row)
                for row in connection.execute(
                    f"SELECT planned_session_id, title, action, duration_seconds "
                    f"FROM workout_prescriptions WHERE status='active' "
                    f"AND planned_session_id IN ({placeholders})", ids
                )
            }
    by_day: dict[str, list[dict]] = {}
    for item in items:
        by_day.setdefault(str(item["planned_date"]), []).append(item)
    cards = []
    for offset in range(7):
        day = monday + timedelta(days=offset)
        sessions = []
        for item in by_day.get(day.isoformat(), []):
            prescribed = prescriptions.get(int(item["id"])) or {}
            state = str(item["completion_state"]).replace("_", " ").title()
            actual = item.get("actual_name") or item.get("workbook_actual", {}).get("Actual Session")
            title = str(prescribed.get("title") or item["title"])
            action = str(prescribed.get("action") or "KEEP").replace("_", " ").title()
            detail = f"{format_duration(prescribed.get('duration_seconds') or item.get('planned_duration_seconds'))} · {action}"
            if actual:
                detail += f" · Done: {actual}"
            sessions.append(
                '<div class="lab-day-session">'
                f'<strong>{escape(sport_marker(item["sport"]))} · {escape(title)}</strong>'
                f'<small>{escape(detail)}</small><small>{escape(state)}'
                f' · {escape(str(item.get("priority") or "No priority"))}</small></div>'
            )
        cards.append(
            f'<div class="lab-day-card {"today" if day == today else ""}">'
            f'<header>{day.strftime("%a")} {day.day:02d}</header>'
            f'<div>{"".join(sessions) if sessions else "<span class=lab-sub>No planned work</span>"}</div></div>'
        )
    st.markdown('<div class="lab-week-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def _target_summary(item: dict) -> str:
    targets = [target["raw_text"] for target in item.get("targets", []) if target.get("raw_text")]
    return "; ".join(targets) if targets else (item.get("description") or "Unavailable")


def _format_distance(value) -> str:
    return f"{float(value) / 1000:.1f} km" if value else "Unavailable"


def _distance_summary(values: dict[str, float]) -> str:
    parts = [f"{sport.title()} {distance / 1000:.1f} km" for sport, distance in values.items() if distance]
    return ", ".join(parts) if parts else "unavailable"


def _open_activity(activity_id: int) -> None:
    st.session_state["selected_activity_id"] = activity_id
    st.session_state["primary_nav"] = "Activities"


def _session_detail(item: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Planned")
        st.markdown(f"**{item['title']}**")
        st.write(item.get("description") or "No detailed prescription.")
        st.caption(
            f"{item['planned_date']} · {item['sport'].title()} · "
            f"{format_duration(item.get('planned_duration_seconds'))} · {item.get('priority') or 'No priority'}"
        )
        for target in item.get("targets", []):
            st.write(f"**{target['target_type'].replace('_', ' ').title()}:** {target['raw_text']} ({target['confidence']} confidence)")
    with right:
        st.markdown("#### Completed")
        actual = item.get("actual_name") or item.get("workbook_actual", {}).get("Actual Session")
        if not actual:
            st.caption("No completed activity is linked.")
        else:
            st.markdown(f"**{actual}**")
            st.write(
                f"Duration: {item['duration_adherence']} · Distance: {item['distance_adherence']} · "
                f"Intensity: {item['intensity_adherence']}"
            )
            if item.get("activity_id"):
                st.caption(f"Activity #{item['activity_id']} · {item.get('match_status')} match · score {item.get('match_score', 0):.0f}")
                st.button(
                    "Open activity detail",
                    key=f"open-activity-{item['id']}",
                    on_click=_open_activity,
                    args=(int(item["activity_id"]),),
                )
            else:
                st.caption("Metadata-only completion from the coaching workbook.")
    reasons = item.get("matching_reason", {}).get("signals", [])
    if reasons:
        st.caption("Match evidence: " + "; ".join(reasons))


def _performance_signals(today: date) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    history = power_history(1200)
    if not history.empty:
        recent = history[history["started_at"] >= pd.Timestamp(today - timedelta(days=42), tz="UTC")]
        if not recent.empty:
            value = recent.iloc[-1]["best_power_w"]
            signals.append({"label": "Recent 20 min power", "value": f"{value:.0f} W", "context": "Best effort in the last 6 weeks"})
    frame = activities(start=today - timedelta(days=28))
    if not frame.empty:
        signals.append({"label": "28-day sessions", "value": str(len(frame)), "context": "Measured completed activities"})
        efficient = frame[frame["efficiency_factor"].notna()]
        if not efficient.empty:
            signals.append({"label": "Recent efficiency", "value": f"{efficient.iloc[0]['efficiency_factor']:.2f}", "context": "Latest activity with sufficient output and HR"})
    return signals[:3]
