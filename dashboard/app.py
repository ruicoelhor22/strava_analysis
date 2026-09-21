from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard._shared import (
    COLORS,
    configure_page,
    date_window,
    empty_state,
    format_duration,
    format_pace,
    page_intro,
    parse_json_list,
    sport_color,
    style_figure,
)
from endurance_lab.analytics import analyze_database
from endurance_lab.config import load_athlete_config, paths
from endurance_lab.quality import ingestion_quality_report
from endurance_lab.queries import (
    activities,
    activity,
    activity_streams,
    comparable_activities,
    daily_load,
    data_quality,
    import_history,
    laps,
    power_curve,
    power_history,
    strength_sets,
)


@st.cache_data(ttl=30, show_spinner=False)
def all_activities() -> pd.DataFrame:
    return activities()


@st.cache_data(ttl=30, show_spinner=False)
def load_series() -> pd.DataFrame:
    return daily_load()


def main() -> None:
    configure_page()
    frame = all_activities()
    with st.sidebar:
        st.markdown("## Endurance Lab")
        st.caption("Private performance analysis")
        page = st.radio(
            "Navigation",
            ["Overview", "Training load", "Cycling", "Running", "Swimming", "Strength", "Activities", "Activity detail", "Data & settings"],
            label_visibility="collapsed",
        )
        st.divider()
        range_choice = st.selectbox(
            "Analysis window", ["7 days", "28 days", "6 weeks", "3 months", "6 months", "All time", "Custom"], index=4
        )
        custom_dates = None
        if range_choice == "Custom":
            today = datetime.now(timezone.utc).date()
            minimum = frame["started_at"].min().date() if not frame.empty else today - timedelta(days=30)
            maximum = frame["started_at"].max().date() if not frame.empty else today
            custom_dates = st.date_input(
                "Custom dates", value=(minimum, maximum), min_value=minimum, max_value=max(maximum, today)
            )
        available_sports = sorted(frame["sport"].dropna().unique().tolist()) if not frame.empty else []
        selected_sports = st.multiselect("Sports", available_sports, default=available_sports)
        config = load_athlete_config()
        ftp_history = config.get("cycling", {}).get("ftp_history", [])
        ftp = ftp_history[-1].get("watts") if ftp_history else None
        st.divider()
        st.caption(f"FTP reference  {ftp or '—'} W")
        st.caption(f"Aerobic HR ceiling  {config.get('athlete', {}).get('aerobic_hr_upper_bpm', '—')} bpm")
        st.markdown('<span class="status-pill">Local only</span>', unsafe_allow_html=True)

    if range_choice == "Custom" and custom_dates and len(custom_dates) == 2:
        start = datetime.combine(custom_dates[0], datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(custom_dates[1] + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    else:
        start, end = date_window(range_choice if range_choice != "Custom" else "All time")
    period_all = _filter_frame(frame, start, end, available_sports)
    filtered = _filter_frame(frame, start, end, selected_sports)
    if frame.empty and page != "Data & settings":
        page_intro("Performance workspace", "Endurance Lab", "Measured data first. Estimates are labelled; unavailable metrics stay unavailable.")
        empty_state()
        return

    if page == "Overview":
        overview_page(filtered, start, end)
    elif page == "Training load":
        training_load_page(start, end)
    elif page == "Cycling":
        cycling_page(filtered[filtered["sport"] == "cycling"])
    elif page == "Running":
        running_page(filtered[filtered["sport"] == "running"])
    elif page == "Swimming":
        swimming_page(filtered[filtered["sport"] == "swimming"])
    elif page == "Strength":
        strength_page(filtered[filtered["sport"] == "strength"], period_all)
    elif page == "Activities":
        activities_page(filtered)
    elif page == "Activity detail":
        activity_detail_page(frame)
    else:
        data_page()


def overview_page(frame: pd.DataFrame, start, end) -> None:
    page_intro("Current block", "Training overview", "Volume, load, balance, and aerobic signals in one compact view.")
    if frame.empty:
        st.info("No activities match the selected date and sport filters.")
        return
    duration = frame["moving_seconds"].fillna(frame["elapsed_seconds"]).fillna(0)
    cycling = frame[frame["sport"] == "cycling"]
    running = frame[frame["sport"] == "running"]
    swimming = frame[frame["sport"] == "swimming"]
    cols = st.columns(6)
    cols[0].metric("Training time", f"{duration.sum() / 3600:.1f} h")
    cols[1].metric("Sessions", f"{len(frame)}")
    cols[2].metric("Cycling", f"{cycling['distance_m'].fillna(0).sum() / 1000:.0f} km")
    cols[3].metric("Running", f"{running['distance_m'].fillna(0).sum() / 1000:.1f} km")
    cols[4].metric("Swimming", f"{swimming['distance_m'].fillna(0).sum() / 1000:.1f} km")
    cols[5].metric("Elevation", f"{frame['ascent_m'].fillna(0).sum():,.0f} m")

    load = _filter_load(load_series(), start, end)
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Training rhythm")
        weekly = _weekly_sport_hours(frame)
        fig = go.Figure()
        for sport in weekly.columns:
            fig.add_bar(x=weekly.index, y=weekly[sport], name=sport.title(), marker_color=sport_color(sport))
        fig.update_layout(barmode="stack", yaxis_title="Hours")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.subheader("Sport distribution")
        distribution = (
            frame.assign(hours=duration / 3600).groupby("sport", as_index=False)["hours"].sum().sort_values("hours", ascending=False)
        )
        fig = go.Figure(go.Bar(
            x=distribution["hours"], y=distribution["sport"].str.title(), orientation="h",
            marker_color=[sport_color(value) for value in distribution["sport"]],
            text=distribution["hours"].map(lambda value: f"{value:.1f}h"), textposition="auto",
        ))
        fig.update_layout(xaxis_title="Hours", yaxis_title=None)
        st.plotly_chart(style_figure(fig), width="stretch")

    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Load · fitness · fatigue")
        if load.empty:
            st.caption("Run analytics to calculate training load.")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=load["day"], y=load["total_load"], name="Daily load", marker_color="#334155", secondary_y=False)
            fig.add_scatter(x=load["day"], y=load["fitness"], name="Fitness", line_color=COLORS["fitness"], secondary_y=True)
            fig.add_scatter(x=load["day"], y=load["fatigue"], name="Fatigue", line_color=COLORS["fatigue"], secondary_y=True)
            fig.add_scatter(x=load["day"], y=load["form"], name="Form", line_color=COLORS["form"], secondary_y=True)
            st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.subheader("Recent aerobic efficiency")
        efficient = frame[frame["efficiency_factor"].notna()].sort_values("started_at")
        if efficient.empty:
            st.caption("Efficiency needs heart rate plus power or speed streams.")
        else:
            fig = go.Figure()
            for sport, group in efficient.groupby("sport"):
                fig.add_scatter(
                    x=group["started_at"], y=group["efficiency_factor"], name=sport.title(),
                    mode="lines+markers", line_color=sport_color(sport), marker_size=5,
                )
            fig.update_layout(yaxis_title="Output / bpm")
            st.plotly_chart(style_figure(fig), width="stretch")

    st.subheader("Recent sessions")
    st.dataframe(_activity_table(frame.head(12)), width="stretch", hide_index=True)


def training_load_page(start, end) -> None:
    page_intro("Adaptation model", "Training load", "Estimated session stress with configurable 42-day fitness and 7-day fatigue time constants.")
    load = _filter_load(load_series(), start, end)
    if load.empty:
        st.info("No calculated load in this window. Run analytics after importing activities.")
        return
    latest = load.iloc[-1]
    cols = st.columns(5)
    cols[0].metric("7-day load", f"{load.tail(7)['total_load'].sum():.0f}")
    cols[1].metric("28-day load", f"{load.tail(28)['total_load'].sum():.0f}")
    cols[2].metric("Fitness", f"{latest['fitness']:.1f}")
    cols[3].metric("Fatigue", f"{latest['fatigue']:.1f}")
    cols[4].metric("Form", f"{latest['form']:+.1f}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=load["day"], y=load["total_load"], name="Daily load", marker_color="#405166", secondary_y=False)
    for name in ("fitness", "fatigue", "form"):
        fig.add_scatter(x=load["day"], y=load[name], name=name.title(), line_color=COLORS[name], secondary_y=True)
    fig.update_yaxes(title_text="Daily estimated load", secondary_y=False)
    fig.update_yaxes(title_text="Modelled load", secondary_y=True)
    st.plotly_chart(style_figure(fig, 410), width="stretch")

    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Weekly load by sport")
        sport_columns = ["cycling_load", "running_load", "swimming_load", "strength_load", "other_load"]
        weekly = load.set_index("day")[sport_columns].resample("W-MON").sum()
        fig = go.Figure()
        for column in sport_columns:
            sport = column.removesuffix("_load")
            fig.add_bar(x=weekly.index, y=weekly[column], name=sport.title(), marker_color=sport_color(sport))
        fig.update_layout(barmode="stack", yaxis_title="Estimated load")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.subheader("Frequency")
        weekly_frequency = load.set_index("day")[["sessions", "hard_sessions", "long_sessions"]].resample("W-MON").sum()
        fig = go.Figure()
        for column, color in (("sessions", "#8FA0B3"), ("hard_sessions", "#F58B8B"), ("long_sessions", "#67B7F7")):
            fig.add_scatter(x=weekly_frequency.index, y=weekly_frequency[column], name=column.replace("_", " ").title(), mode="lines+markers", line_color=color)
        st.plotly_chart(style_figure(fig), width="stretch")
    st.caption("All load values are estimates. Power TSS is preferred for cycling, then Edwards HR-zone load, then a labelled duration-only fallback.")


def cycling_page(frame: pd.DataFrame) -> None:
    page_intro("Cycling performance", "Power & aerobic durability", "Power-duration development, efficiency, drift, and long-ride execution.")
    if frame.empty:
        st.info("No cycling activities match the current filters.")
        return
    powered = frame[frame["avg_power_w"].notna()]
    cols = st.columns(5)
    cols[0].metric("Volume", f"{frame['moving_seconds'].fillna(0).sum() / 3600:.1f} h")
    cols[1].metric("Distance", f"{frame['distance_m'].fillna(0).sum() / 1000:.0f} km")
    cols[2].metric("Powered rides", f"{len(powered)}")
    cols[3].metric("Best 20 min", _best_power_label(1200))
    applicable = frame[frame["decoupling_status"] == "applicable"]
    cols[4].metric("Median drift", "—" if applicable.empty else f"{applicable['aerobic_decoupling_pct'].median():.1f}%")

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Power-duration curve")
        historical = power_curve()
        recent = power_curve(days=42)
        if historical.empty:
            st.caption("Power streams are needed for a power-duration curve.")
        else:
            fig = go.Figure()
            fig.add_scatter(x=historical["duration_seconds"], y=historical["best_power_w"], name="All time", mode="lines+markers", line_color="#8190A5")
            if not recent.empty:
                fig.add_scatter(x=recent["duration_seconds"], y=recent["best_power_w"], name="Last 6 weeks", mode="lines+markers", line_color=COLORS["cycling"])
            fig.update_xaxes(type="log", tickvals=[5, 15, 30, 60, 120, 300, 600, 1200, 3600], ticktext=["5s", "15s", "30s", "1m", "2m", "5m", "10m", "20m", "60m"])
            fig.update_layout(yaxis_title="Watts", xaxis_title="Duration")
            st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.subheader("20-minute power history")
        history = power_history(1200)
        if history.empty:
            st.caption("No 20-minute efforts available.")
        else:
            fig = go.Figure(go.Scatter(x=history["started_at"], y=history["best_power_w"], mode="lines+markers", line_color=COLORS["cycling"], marker_size=6))
            fig.update_layout(yaxis_title="Watts")
            st.plotly_chart(style_figure(fig), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Power / HR efficiency")
        _metric_trend(frame, "efficiency_factor", "W per bpm", COLORS["cycling"])
    with right:
        st.subheader("Aerobic decoupling")
        drift = frame[frame["decoupling_status"] == "applicable"].sort_values("started_at")
        if drift.empty:
            st.caption("No steady, sufficiently long rides qualify for decoupling analysis.")
        else:
            fig = go.Figure(go.Scatter(x=drift["started_at"], y=drift["aerobic_decoupling_pct"], mode="markers+lines", line_color="#F58B8B"))
            fig.add_hline(y=5, line_dash="dot", line_color="#6B7280")
            fig.update_layout(yaxis_title="Decoupling %")
            st.plotly_chart(style_figure(fig), width="stretch")
    st.subheader("Ride inventory")
    st.dataframe(_activity_table(frame), width="stretch", hide_index=True)


def running_page(frame: pd.DataFrame) -> None:
    page_intro("Running performance", "Pace & heart-rate economy", "Weekly volume, aerobic efficiency, long-run durability, and comparable efforts.")
    if frame.empty:
        st.info("No running activities match the current filters.")
        return
    cols = st.columns(5)
    cols[0].metric("Volume", f"{frame['distance_m'].fillna(0).sum() / 1000:.1f} km")
    cols[1].metric("Time", f"{frame['moving_seconds'].fillna(0).sum() / 3600:.1f} h")
    cols[2].metric("Sessions", f"{len(frame)}")
    pace_values = frame["pace_seconds_per_km"].dropna()
    cols[3].metric("Median pace", format_pace(pace_values.median() if not pace_values.empty else None))
    drift = frame[frame["decoupling_status"] == "applicable"]
    cols[4].metric("Median drift", "—" if drift.empty else f"{drift['aerobic_decoupling_pct'].median():.1f}%")

    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Weekly running volume")
        tmp = frame.copy()
        tmp["week"] = tmp["started_at"].dt.tz_convert(None).dt.to_period("W-MON").dt.start_time
        weekly = tmp.groupby("week").agg(distance_m=("distance_m", "sum"), duration=("moving_seconds", "sum")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=weekly["week"], y=weekly["distance_m"] / 1000, name="Distance", marker_color=COLORS["running"], secondary_y=False)
        fig.add_scatter(x=weekly["week"], y=weekly["duration"] / 3600, name="Hours", line_color="#67B7F7", secondary_y=True)
        fig.update_yaxes(title_text="Kilometres", secondary_y=False)
        fig.update_yaxes(title_text="Hours", secondary_y=True)
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.subheader("Pace relative to HR")
        usable = frame[frame["avg_hr"].notna() & frame["pace_seconds_per_km"].notna()].copy()
        if usable.empty:
            st.caption("Pace/HR needs both distance and heart-rate data.")
        else:
            usable["pace_min_km"] = usable["pace_seconds_per_km"] / 60
            fig = go.Figure(go.Scatter(
                x=usable["avg_hr"], y=usable["pace_min_km"], mode="markers",
                marker=dict(color=usable["started_at"].astype("int64"), colorscale="Viridis", size=8, showscale=False),
                text=usable["started_at"].dt.strftime("%d %b %Y"),
            ))
            fig.update_yaxes(autorange="reversed", title="Pace min/km")
            fig.update_xaxes(title="Average HR bpm")
            st.plotly_chart(style_figure(fig), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Speed / HR efficiency")
        _metric_trend(frame, "efficiency_factor", "m/s per bpm", COLORS["running"])
    with right:
        st.subheader("Long-run pace retention")
        long_runs = frame[frame["moving_seconds"].fillna(0) >= 4500].sort_values("started_at")
        if long_runs.empty or long_runs["late_fade_pct"].dropna().empty:
            st.caption("No long runs with sufficient stream data in this window.")
        else:
            fig = go.Figure(go.Scatter(x=long_runs["started_at"], y=long_runs["late_fade_pct"], mode="lines+markers", line_color="#F58B8B"))
            fig.add_hline(y=0, line_color="#6B7280")
            fig.update_layout(yaxis_title="Second-half speed fade %")
            st.plotly_chart(style_figure(fig), width="stretch")
    st.subheader("Run inventory")
    st.dataframe(_activity_table(frame), width="stretch", hide_index=True)


def swimming_page(frame: pd.DataFrame) -> None:
    page_intro("Swimming performance", "Pace & retention", "Reliable distance, duration, lap consistency, and late-session pace retention when available.")
    if frame.empty:
        st.info("No swimming activities match the current filters.")
        return
    duration = frame["moving_seconds"].fillna(frame["elapsed_seconds"]).fillna(0)
    distance = frame["distance_m"].fillna(0)
    pace_100 = duration / (distance / 100).replace(0, math.nan)
    cols = st.columns(5)
    cols[0].metric("Sessions", f"{len(frame)}")
    cols[1].metric("Distance", f"{distance.sum() / 1000:.1f} km")
    cols[2].metric("Time", f"{duration.sum() / 3600:.1f} h")
    cols[3].metric("Median pace", _swim_pace(pace_100.median() if pace_100.notna().any() else None))
    fade = pd.to_numeric(frame["late_fade_pct"], errors="coerce").dropna()
    cols[4].metric("Median late fade", "—" if fade.empty else f"{fade.median():.1f}%")
    plotted = frame.copy().sort_values("started_at")
    plotted["pace_100"] = (
        plotted["moving_seconds"].fillna(plotted["elapsed_seconds"]) / (plotted["distance_m"] / 100).replace(0, math.nan)
    )
    st.subheader("Pace trend")
    if plotted["pace_100"].notna().any():
        fig = go.Figure(go.Scatter(x=plotted["started_at"], y=plotted["pace_100"], mode="lines+markers", line_color=COLORS["swimming"]))
        fig.update_yaxes(autorange="reversed", title="Seconds / 100 m")
        st.plotly_chart(style_figure(fig), width="stretch")
    else:
        st.caption("Swimming pace requires both distance and duration.")
    st.subheader("Swim inventory")
    st.dataframe(_activity_table(frame), width="stretch", hide_index=True)


def strength_page(frame: pd.DataFrame, period_all: pd.DataFrame) -> None:
    page_intro("Strength support", "Strength frequency & proximity", "Session timing and imported set details, without inventing missing units or measurements.")
    if frame.empty:
        st.info("No strength activities match the current filters.")
        return
    duration = frame["moving_seconds"].fillna(frame["elapsed_seconds"]).fillna(0)
    span_weeks = max(1.0, (frame["started_at"].max() - frame["started_at"].min()).days / 7)
    endurance = period_all[period_all["sport"].isin(["cycling", "running", "swimming"])]
    proximity = _strength_proximity(frame, endurance)
    cols = st.columns(4)
    cols[0].metric("Sessions", f"{len(frame)}")
    cols[1].metric("Time", f"{duration.sum() / 3600:.1f} h")
    cols[2].metric("Frequency", f"{len(frame) / span_weeks:.1f} / week")
    cols[3].metric("Within 24h endurance", f"{int((proximity['nearest endurance h'] <= 24).sum()) if not proximity.empty else 0}")
    tmp = frame.copy()
    tmp["week"] = tmp["started_at"].dt.tz_convert(None).dt.to_period("W-MON").dt.start_time
    weekly = tmp.groupby("week").agg(sessions=("id", "count"), duration=("moving_seconds", "sum")).reset_index()
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Weekly consistency")
        fig = go.Figure(go.Bar(x=weekly["week"], y=weekly["sessions"], marker_color=COLORS["strength"]))
        fig.update_layout(yaxis_title="Sessions")
        st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.subheader("Nearest endurance session")
        if proximity.empty or proximity["nearest endurance h"].isna().all():
            st.caption("No endurance sessions available in this window.")
        else:
            fig = go.Figure(go.Histogram(x=proximity["nearest endurance h"], nbinsx=12, marker_color="#67B7F7"))
            fig.update_layout(xaxis_title="Hours", yaxis_title="Strength sessions")
            st.plotly_chart(style_figure(fig), width="stretch")
    st.subheader("Session proximity")
    st.dataframe(proximity, width="stretch", hide_index=True)
    set_frame = strength_sets(frame["id"].astype(int).tolist())
    st.subheader("Exercises & sets")
    if set_frame.empty:
        st.caption("No exercise-level set data was present in the imported source files.")
    else:
        display = set_frame[["activity_started_at", "exercise_name", "set_number", "repetitions", "load_value", "load_unit"]].copy()
        display["activity_started_at"] = pd.to_datetime(display["activity_started_at"], utc=True).dt.strftime("%d %b %Y")
        display["exercise_name"] = display["exercise_name"].str.replace("_", " ").str.title()
        st.dataframe(display, width="stretch", hide_index=True)
        if display["load_unit"].isna().any():
            st.caption("The supplied strength JSON contains load numbers but no weight unit; values are preserved without assuming kg or lb.")


def activities_page(frame: pd.DataFrame) -> None:
    page_intro("Session library", "Activities", "Search the normalized record; open Activity detail for full-resolution analysis.")
    search = st.text_input("Search by name or source file", placeholder="Long ride, morning run…")
    shown = frame
    if search:
        mask = frame["name"].fillna("").str.contains(search, case=False) | frame["source_filename"].fillna("").str.contains(search, case=False)
        shown = frame[mask]
    st.caption(f"{len(shown):,} activities")
    st.dataframe(_activity_table(shown), width="stretch", hide_index=True, height=620)


def activity_detail_page(frame: pd.DataFrame) -> None:
    page_intro("Session analysis", "Activity detail", "Aligned streams, laps, zones, halves, drift, best efforts, and comparable sessions.")
    options = {
        f"{row.started_at.strftime('%d %b %Y')} · {str(row.sport).title()} · {row.name or 'Untitled'}": int(row.id)
        for row in frame.sort_values("started_at", ascending=False).itertuples()
    }
    selected_label = st.selectbox("Activity", list(options))
    record = activity(options[selected_label])
    if not record:
        st.error("Activity not found.")
        return
    activity_id = int(record["id"])
    duration = record.get("moving_seconds") or record.get("elapsed_seconds")
    cols = st.columns(6)
    cols[0].metric("Duration", format_duration(duration))
    cols[1].metric("Distance", "—" if not record.get("distance_m") else f"{record['distance_m'] / 1000:.1f} km")
    cols[2].metric("Elevation", "—" if record.get("ascent_m") is None else f"{record['ascent_m']:.0f} m")
    cols[3].metric("Avg HR", "—" if record.get("avg_hr") is None else f"{record['avg_hr']:.0f} bpm")
    if record["sport"] == "cycling":
        cols[4].metric("Normalised power", "—" if record.get("normalized_power_w") is None else f"{record['normalized_power_w']:.0f} W")
    else:
        cols[4].metric("Pace", format_pace(record.get("pace_seconds_per_km")))
    cols[5].metric("Estimated load", "—" if record.get("selected_load") is None else f"{record['selected_load']:.0f}")
    st.caption(f"Load method: {(record.get('load_method') or 'not calculated').replace('_', ' ')} · Source: {record['source_filename']}")

    streams = activity_streams(activity_id)
    if streams.empty:
        st.info("No time-series trackpoints were present in this activity source.")
    else:
        streams["minutes"] = streams["elapsed_seconds"] / 60
        output_chart = make_subplots(specs=[[{"secondary_y": True}]])
        if record["sport"] == "cycling" and streams["power_w"].notna().any():
            output_chart.add_scatter(x=streams["minutes"], y=streams["power_w"], name="Power", line_color=COLORS["cycling"], secondary_y=False)
            output_title = "Watts"
        elif streams["speed_mps"].notna().any():
            pace = 1000 / streams["speed_mps"].replace(0, math.nan) / 60
            output_chart.add_scatter(x=streams["minutes"], y=pace, name="Pace", line_color=COLORS["running"], secondary_y=False)
            output_chart.update_yaxes(autorange="reversed", secondary_y=False)
            output_title = "Pace min/km"
        else:
            output_title = "Output"
        if streams["heart_rate"].notna().any():
            output_chart.add_scatter(x=streams["minutes"], y=streams["heart_rate"], name="Heart rate", line_color="#F58B8B", secondary_y=True)
        output_chart.update_yaxes(title_text=output_title, secondary_y=False)
        output_chart.update_yaxes(title_text="Heart rate bpm", secondary_y=True)
        output_chart.update_xaxes(title="Elapsed minutes")
        st.subheader("Output & heart rate")
        st.plotly_chart(style_figure(output_chart, 390), width="stretch")

        left, right = st.columns(2)
        with left:
            st.subheader("Cadence")
            if streams["cadence"].notna().any():
                fig = go.Figure(go.Scatter(x=streams["minutes"], y=streams["cadence"], mode="lines", line_color="#C7A0FF"))
                fig.update_layout(yaxis_title="rpm / spm", xaxis_title="Elapsed minutes")
                st.plotly_chart(style_figure(fig, 260), width="stretch")
            else:
                st.caption("Cadence unavailable.")
        with right:
            st.subheader("Elevation")
            if streams["altitude_m"].notna().any():
                fig = go.Figure(go.Scatter(x=streams["minutes"], y=streams["altitude_m"], fill="tozeroy", line_color="#67B7F7"))
                fig.update_layout(yaxis_title="Metres", xaxis_title="Elapsed minutes")
                st.plotly_chart(style_figure(fig, 260), width="stretch")
            else:
                st.caption("Elevation unavailable.")

    st.subheader("Execution")
    exec_cols = st.columns(5)
    exec_cols[0].metric("First-half output", _output_label(record.get("first_half_output"), record["sport"]))
    exec_cols[1].metric("Second-half output", _output_label(record.get("second_half_output"), record["sport"]))
    exec_cols[2].metric("First-half HR", _number_label(record.get("first_half_hr"), "bpm"))
    exec_cols[3].metric("Second-half HR", _number_label(record.get("second_half_hr"), "bpm"))
    exec_cols[4].metric("Aerobic decoupling", "—" if record.get("aerobic_decoupling_pct") is None else f"{record['aerobic_decoupling_pct']:.1f}%")
    if record.get("decoupling_status") != "applicable":
        st.caption(f"Decoupling not applicable: {(record.get('decoupling_reason') or 'insufficient data').replace('_', ' ')}.")

    left, right = st.columns(2)
    with left:
        _zones_panel(record, "hr_zones_json", "Heart-rate zones", "#F58B8B")
    with right:
        _zones_panel(record, "power_zones_json", "Power zones", COLORS["cycling"])

    lap_frame = laps(activity_id)
    if not lap_frame.empty:
        st.subheader("Laps / splits")
        display = lap_frame[["lap_number", "duration_seconds", "distance_m", "avg_hr", "avg_power_w", "avg_cadence"]].copy()
        display["duration"] = display["duration_seconds"].map(format_duration)
        display["distance km"] = display["distance_m"] / 1000
        display = display.rename(columns={"lap_number": "lap", "avg_hr": "avg HR", "avg_power_w": "avg power", "avg_cadence": "avg cadence"})
        st.dataframe(display[["lap", "duration", "distance km", "avg HR", "avg power", "avg cadence"]], width="stretch", hide_index=True)

    set_frame = strength_sets([activity_id])
    if not set_frame.empty:
        st.subheader("Strength sets")
        display_sets = set_frame[["exercise_name", "set_number", "repetitions", "load_value", "load_unit"]].copy()
        display_sets["exercise_name"] = display_sets["exercise_name"].str.replace("_", " ").str.title()
        st.dataframe(display_sets, width="stretch", hide_index=True)

    comparable = comparable_activities(activity_id)
    st.subheader("Comparable sessions · previous 8 weeks")
    if comparable.empty:
        st.caption("No prior sessions passed the sport, duration, distance, and elevation similarity filters.")
    else:
        target_efficiency = record.get("efficiency_factor")
        median_efficiency = comparable["efficiency_factor"].dropna().median()
        if target_efficiency and pd.notna(median_efficiency) and median_efficiency:
            delta = (target_efficiency / median_efficiency - 1) * 100
            st.metric("Efficiency vs comparable median", f"{delta:+.1f}%", help=f"Based on {len(comparable)} similar prior sessions")
        st.dataframe(_activity_table(comparable), width="stretch", hide_index=True)


def data_page() -> None:
    page_intro("Local system", "Data & settings", "Import health, private storage, athlete thresholds, and recalculation controls.")
    quality = data_quality()
    cols = st.columns(5)
    cols[0].metric("Activities", f"{quality.get('activities', 0):,}")
    cols[1].metric("Trackpoints", f"{quality.get('trackpoints', 0):,}")
    cols[2].metric("Missing HR", f"{quality.get('missing_hr', 0):,}")
    cols[3].metric("Cycling without power", f"{quality.get('cycling_missing_power', 0):,}")
    cols[4].metric("Missing distance", f"{quality.get('missing_distance', 0):,}")
    ingestion = ingestion_quality_report()
    report_left, report_right = st.columns(2)
    with report_left:
        st.subheader("Activities by sport")
        st.dataframe(
            pd.DataFrame(
                [{"sport": key.title(), "activities": value} for key, value in ingestion["activities_by_sport"].items()]
            ),
            width="stretch",
            hide_index=True,
        )
    with report_right:
        st.subheader("Primary source format")
        st.dataframe(
            pd.DataFrame(
                [{"format": key.upper(), "activities": value} for key, value in ingestion["activities_by_source_format"].items()]
            ),
            width="stretch",
            hide_index=True,
        )
    st.subheader("Sensor and structure coverage")
    coverage = ingestion["activities_containing"]
    st.dataframe(
        pd.DataFrame([{"field": key.replace("_", " ").title(), "activities": value} for key, value in coverage.items()]),
        width="stretch",
        hide_index=True,
    )
    if ingestion["suspicious_activities"]:
        st.subheader("Suspicious activities")
        suspicious = pd.DataFrame(ingestion["suspicious_activities"])
        suspicious["flags"] = suspicious["flags"].map(lambda value: ", ".join(value))
        st.dataframe(suspicious, width="stretch", hide_index=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Import history")
        history = import_history()
        if history.empty:
            st.caption("No import attempts yet.")
        else:
            st.dataframe(history, width="stretch", hide_index=True)
    with right:
        st.subheader("Local paths")
        local_paths = paths()
        st.code(f"Database\n{local_paths.database}\n\nActivity inbox\n{local_paths.import_dir}\n\nRaw archive\n{local_paths.raw_dir}")
        if st.button("Recalculate all analytics", type="primary", width="stretch"):
            with st.spinner("Recalculating from normalized streams…"):
                count = analyze_database()
                st.cache_data.clear()
            st.success(f"Recalculated {count} activities.")
    st.subheader("Athlete configuration")
    st.json(load_athlete_config(), expanded=False)
    st.warning("This dashboard is intentionally bound to localhost by default. Raw activity files, GPS tracks, databases, and private athlete configuration are Git-ignored.")


def _filter_frame(frame: pd.DataFrame, start, end, sports: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["started_at"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["started_at"] < pd.Timestamp(end)
    if sports:
        mask &= frame["sport"].isin(sports)
    return frame[mask].copy()


def _filter_load(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["day"] >= pd.Timestamp(start.date())
    if end is not None:
        mask &= frame["day"] < pd.Timestamp(end.date())
    return frame[mask].copy()


def _weekly_sport_hours(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame.copy()
    tmp["week"] = tmp["started_at"].dt.tz_convert(None).dt.to_period("W-MON").dt.start_time
    tmp["hours"] = tmp["moving_seconds"].fillna(tmp["elapsed_seconds"]).fillna(0) / 3600
    return tmp.pivot_table(index="week", columns="sport", values="hours", aggfunc="sum", fill_value=0)


def _activity_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = pd.DataFrame()
    display["date"] = frame["started_at"].dt.strftime("%d %b %Y")
    display["sport"] = frame["sport"].str.title()
    display["source"] = frame["source_format"].str.upper()
    display["session"] = frame["name"].fillna("Untitled")
    display["duration"] = frame["moving_seconds"].fillna(frame["elapsed_seconds"]).map(format_duration)
    display["distance km"] = (pd.to_numeric(frame["distance_m"], errors="coerce") / 1000).round(1)
    display["avg HR"] = pd.to_numeric(frame["avg_hr"], errors="coerce").round(0)
    display["avg power"] = pd.to_numeric(frame["avg_power_w"], errors="coerce").round(0)
    display["pace"] = frame["pace_seconds_per_km"].map(format_pace)
    display["load est."] = pd.to_numeric(frame["selected_load"], errors="coerce").round(0)
    display["drift %"] = pd.to_numeric(frame["aerobic_decoupling_pct"], errors="coerce").round(1)
    return display


def _metric_trend(frame: pd.DataFrame, column: str, label: str, color: str) -> None:
    usable = frame[frame[column].notna()].sort_values("started_at")
    if usable.empty:
        st.caption("Not enough matching stream data.")
        return
    usable = usable.copy()
    usable["rolling"] = usable[column].rolling(5, min_periods=2).median()
    fig = go.Figure()
    fig.add_scatter(x=usable["started_at"], y=usable[column], mode="markers", name="Session", marker_color=color, marker_size=6)
    fig.add_scatter(x=usable["started_at"], y=usable["rolling"], mode="lines", name="5-session median", line_color="#EAF0F5")
    fig.update_layout(yaxis_title=label)
    st.plotly_chart(style_figure(fig), width="stretch")


def _zones_panel(record: dict, key: str, title: str, color: str) -> None:
    st.subheader(title)
    zones = parse_json_list(record.get(key))
    if not zones or sum(zones) <= 0:
        st.caption("Zone distribution unavailable.")
        return
    fig = go.Figure(go.Bar(x=[f"Z{index + 1}" for index in range(len(zones))], y=[value / 60 for value in zones], marker_color=color))
    fig.update_layout(yaxis_title="Minutes")
    st.plotly_chart(style_figure(fig, 260), width="stretch")


def _best_power_label(duration: int) -> str:
    curve = power_curve()
    match = curve[curve["duration_seconds"] == duration]
    return "—" if match.empty else f"{match.iloc[0]['best_power_w']:.0f} W"


def _output_label(value, sport: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.0f} W" if sport == "cycling" else f"{value:.2f} m/s"


def _number_label(value, suffix: str) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.0f} {suffix}"


def _swim_pace(seconds_per_100: float | None) -> str:
    if seconds_per_100 is None or pd.isna(seconds_per_100) or seconds_per_100 <= 0:
        return "—"
    minutes, seconds = divmod(int(round(seconds_per_100)), 60)
    return f"{minutes}:{seconds:02d}/100m"


def _strength_proximity(strength: pd.DataFrame, endurance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session in strength.sort_values("started_at", ascending=False).itertuples():
        nearest_hours = math.nan
        nearest_sport = "—"
        nearest_date = "—"
        if not endurance.empty:
            differences = (endurance["started_at"] - session.started_at).abs()
            closest_index = differences.idxmin()
            nearest_hours = differences.loc[closest_index].total_seconds() / 3600
            nearest_sport = str(endurance.loc[closest_index, "sport"]).title()
            nearest_date = endurance.loc[closest_index, "started_at"].strftime("%d %b %Y %H:%M")
        rows.append(
            {
                "date": session.started_at.strftime("%d %b %Y"),
                "session": session.name or "Strength session",
                "duration": format_duration(session.moving_seconds or session.elapsed_seconds),
                "nearest endurance h": nearest_hours,
                "nearest sport": nearest_sport,
                "nearest session": nearest_date,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["nearest endurance h"] = pd.to_numeric(result["nearest endurance h"], errors="coerce").round(1)
    return result


if __name__ == "__main__":
    main()

