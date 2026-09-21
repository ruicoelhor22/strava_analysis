from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
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
    "other": "#8190A5",
    "fitness": "#73E0A9",
    "fatigue": "#F58B8B",
    "form": "#67B7F7",
    "load": "#94A3B8",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Endurance Lab",
        page_icon="↗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root { --panel:#111923; --border:#253241; --muted:#8fa0b3; --ink:#eaf0f5; --accent:#73e0a9; }
        .stApp { background:#0a1017; color:var(--ink); }
        [data-testid="stSidebar"] { background:#0d141d; border-right:1px solid var(--border); }
        [data-testid="stHeader"] { background:transparent; }
        .block-container { max-width:1540px; padding-top:1.35rem; padding-bottom:3rem; }
        h1 { letter-spacing:-0.045em !important; font-size:2rem !important; margin-bottom:.15rem !important; }
        h2 { letter-spacing:-0.025em !important; font-size:1.18rem !important; }
        h3 { color:#b9c7d5 !important; font-size:.78rem !important; text-transform:uppercase; letter-spacing:.09em; }
        p, label, [data-testid="stCaptionContainer"] { color:var(--muted); }
        [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:.7rem .8rem; }
        [data-testid="stMetricLabel"] { text-transform:uppercase; letter-spacing:.07em; font-size:.7rem; }
        [data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; letter-spacing:-.035em; }
        [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border:1px solid var(--border); border-radius:8px; overflow:hidden; }
        .status-pill { display:inline-block; padding:.18rem .5rem; border:1px solid #335044; color:#8ce8b8; border-radius:999px; font-size:.72rem; }
        .eyebrow { color:#73e0a9; text-transform:uppercase; letter-spacing:.14em; font-weight:700; font-size:.7rem; }
        .context { color:#8fa0b3; margin-top:-.25rem; margin-bottom:1.1rem; }
        div.stButton > button { border-radius:6px; border-color:#314052; }
        hr { border-color:var(--border) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, context: str) -> None:
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="context">{context}</div>', unsafe_allow_html=True)


def style_figure(figure: go.Figure, height: int = 330) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=14, r=14, t=28, b=14),
        paper_bgcolor="#111923",
        plot_bgcolor="#111923",
        font=dict(color="#9fafbf", size=11),
        hoverlabel=dict(bgcolor="#172230", font_color="#eef4f8"),
        legend=dict(orientation="h", y=1.08, x=0, title=None),
        xaxis=dict(gridcolor="#1e2a37", zeroline=False),
        yaxis=dict(gridcolor="#1e2a37", zeroline=False),
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


def sport_color(sport: str) -> str:
    return COLORS.get(sport, COLORS["other"])


def empty_state() -> None:
    st.markdown(
        """
        <div style="border:1px solid #253241;border-radius:9px;background:#111923;padding:1.2rem 1.3rem;max-width:780px">
          <div class="eyebrow">Ready for private data</div>
          <h2 style="margin:.35rem 0">No activities imported yet</h2>
          <p>Place TCX, FIT, or supported strength JSON files in <code>data/import</code>, run
          <code>python -m endurance_lab rebuild</code>, then refresh this page.</p>
          <p style="margin-bottom:0">Raw files, GPS tracks, and the local database are excluded from Git.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_json_list(value: object) -> list[float]:
    import json

    try:
        return [float(item) for item in json.loads(str(value or "[]"))]
    except (ValueError, TypeError, json.JSONDecodeError):
        return []
