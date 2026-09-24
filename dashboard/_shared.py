from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

COLORS = {
    "cycling": "#73E0A9",
    "running": "#FCB56B",
    "swimming": "#67B7F7",
    "strength": "#C7A0FF",
    "hiking": "#D9C17F",
    "walking": "#D9C17F",
    "other": "#A4B3C1",
    "fitness": "#73E0A9",
    "fatigue": "#F58B8B",
    "form": "#67B7F7",
    "load": "#94A3B8",
}
STATE_COLORS = {
    "good": ("#73E0A9", "rgba(115,224,169,.10)", "#315845"),
    "watch": ("#F7C873", "rgba(247,200,115,.10)", "#66532E"),
    "attention": ("#F58B8B", "rgba(245,139,139,.10)", "#623840"),
    "neutral": ("#AFC0D0", "rgba(143,160,179,.09)", "#344354"),
}
SPORT_MARKERS = {
    "cycling": "BIKE",
    "running": "RUN",
    "swimming": "SWIM",
    "strength": "GYM",
    "hiking": "HIKE",
    "walking": "WALK",
    "other": "OTHER",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Endurance Lab", page_icon="↗", layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        "<style>" + (ROOT / "dashboard" / "styles.css").read_text(encoding="utf-8") + "</style>",
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, context: str) -> None:
    st.markdown(f'<div class="eyebrow">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="context">{escape(context)}</div>', unsafe_allow_html=True)


def section_header(title: str, context: str | None = None) -> None:
    st.markdown(f'<div class="section-label">{escape(title)}</div>', unsafe_allow_html=True)
    if context:
        st.markdown(f'<div class="section-context">{escape(context)}</div>', unsafe_allow_html=True)


def decision_pill(label: str, tone: str = "neutral") -> str:
    foreground, background, border = STATE_COLORS.get(tone, STATE_COLORS["neutral"])
    return (
        f'<span class="decision-pill" style="color:{foreground};background:{background};'
        f'border:1px solid {border}">{escape(str(label))}</span>'
    )


def status_badge(label: str, tone: str = "neutral") -> str:
    foreground = STATE_COLORS.get(tone, STATE_COLORS["neutral"])[0]
    return f'<span class="lab-badge" style="color:{foreground}">{escape(str(label))}</span>'


def sport_color(sport: str) -> str:
    return COLORS.get(str(sport).lower(), COLORS["other"])


def sport_marker(sport: str) -> str:
    return SPORT_MARKERS.get(str(sport).lower(), "OTHER")


def metric_tile(label: str, value: str, context: str = "", tone: str = "neutral") -> str:
    color = STATE_COLORS.get(tone, STATE_COLORS["neutral"])[0]
    return (
        '<div class="lab-state">'
        f'<small>{escape(label)}</small><strong style="color:{color}">{escape(str(value))}</strong>'
        f'<em>{escape(context)}</em></div>'
    )


def cost_indicator(cost: dict | None) -> str:
    if not cost:
        return '<span class="lab-sub">Training cost unavailable</span>'
    labels = (("cardiovascular_cost", "CARDIO"), ("muscular_cost", "MUSCULAR"),
              ("intensity_cost", "INTENSITY"))
    levels = {"low": (25, "#73e0a9"), "moderate": (58, "#f7c873"), "high": (100, "#f58b8b")}
    pieces = []
    for key, label in labels:
        value = str(cost.get(key) or "unknown").lower()
        width, color = levels.get(value, (0, "#a4b3c1"))
        pieces.append(
            f'<div class="lab-cost"><span>{label}</span><div class="lab-cost-track">'
            f'<div class="lab-cost-fill" style="--cost-width:{width}%;--cost-color:{color}"></div></div>'
            f'<span class="lab-cost-value">{escape(value)}</span></div>'
        )
    muscle = cost.get("muscle_load")
    if muscle and muscle != "none":
        pieces.append(f'<div class="lab-sub">{escape(str(muscle).title())} body load</div>')
    return "".join(pieces)


def activity_card(row: dict, cost: dict | None = None) -> str:
    sport = str(row.get("sport") or "other")
    started = pd.to_datetime(row.get("started_at"), utc=True, errors="coerce")
    when = started.strftime("%a %d %b · %H:%M") if pd.notna(started) else "Date unavailable"
    title = str(row.get("name") or sport.title())
    if title == str(row.get("source_activity_id") or ""):
        title = sport.title()
    facts = [format_duration(row.get("moving_seconds") or row.get("elapsed_seconds"))]
    distance = row.get("distance_m")
    if pd.notna(distance) and distance:
        facts.append(f"{float(distance) / 1000:.1f} km")
    if sport == "cycling" and pd.notna(row.get("avg_power_w")):
        facts.append(f"{float(row['avg_power_w']):.0f} W")
    elif sport == "running" and pd.notna(row.get("pace_seconds_per_km")):
        facts.append(format_pace(row["pace_seconds_per_km"]))
    if pd.notna(row.get("avg_hr")):
        facts.append(f"{float(row['avg_hr']):.0f} bpm")
    cost_text = str(cost.get("cost_class") or "").replace("_", " ").upper() if cost else ""
    color = sport_color(sport)
    return (
        f'<div class="lab-row" style="--sport-color:{color}">'
        f'<div class="lab-row-head"><span>{escape(sport_marker(sport))} · {escape(when)}</span>'
        f'<span>{escape(cost_text)}</span></div>'
        f'<div class="lab-row-title">{escape(title)}</div>'
        f'<div class="lab-facts">{"".join(f"<span>{escape(str(fact))}</span>" for fact in facts)}</div>'
        '</div>'
    )


def style_figure(figure: go.Figure, height: int = 320) -> go.Figure:
    figure.update_layout(
        height=height, autosize=True,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="#101a24", plot_bgcolor="#101a24",
        font=dict(color="#a4b3c1", size=11),
        hoverlabel=dict(bgcolor="#172230", font_color="#eef4f8"),
        legend=dict(orientation="h", y=1.08, x=0, title=None),
        xaxis=dict(gridcolor="#22303e", zeroline=False, automargin=True),
        yaxis=dict(gridcolor="#22303e", zeroline=False, automargin=True),
    )
    return figure


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def format_pace(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds) or seconds <= 0:
        return "—"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}:{remainder:02d}/km"


def date_window(choice: str) -> tuple[datetime | None, datetime | None]:
    if choice == "All time":
        return None, None
    days = {"7 days": 7, "28 days": 28, "6 weeks": 42, "3 months": 90, "6 months": 183}[choice]
    end = datetime.now(timezone.utc) + timedelta(days=1)
    return end - timedelta(days=days), end


def empty_state() -> None:
    st.info("No activities imported yet. Import FIT, TCX or supported strength files to populate the dashboard.")


def parse_json_list(value: object) -> list[float]:
    try:
        return [float(item) for item in json.loads(str(value or "[]"))]
    except (ValueError, TypeError, json.JSONDecodeError):
        return []


def zone_range_labels(boundaries: list[float], unit: str, *, open_last: bool = False) -> list[str]:
    """Describe the same half-open zones used by analytics for integer sensor readings."""
    if len(boundaries) < 2:
        return []
    labels = []
    for index, (lower, upper) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        start = math.ceil(lower)
        end = math.ceil(upper) - 1
        interval = f"{start}+" if open_last and index == len(boundaries) - 1 else f"{start}–{end}"
        labels.append(f"Z{index} · {interval} {unit}")
    return labels
