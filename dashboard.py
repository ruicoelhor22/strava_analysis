import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/strava_activities.parquet"

load_dotenv()  # only needed if one day you want to re-use API keys here


# ============================================================
# DATA LOADING & PREP
# ============================================================

@st.cache_data(show_spinner=True)
def load_activities(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Parquet file not found at {path}. Run the extraction script first."
        )
    df = pd.read_parquet(path)
    return prepare_dataframe(df)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime
    df["start_date_local"] = pd.to_datetime(df["start_date_local"], errors="coerce")

    # Basic date parts
    df["date"] = df["start_date_local"].dt.date
    df["year"] = df["start_date_local"].dt.year
    df["month"] = df["start_date_local"].dt.month
    df["week"] = df["start_date_local"].dt.isocalendar().week.astype("Int64")
    df["weekday"] = df["start_date_local"].dt.weekday  # Monday=0, Sunday=6

    # Distance: meters -> km
    if "distance" in df.columns:
        df["distance_km"] = df["distance"].fillna(0) / 1000.0
    else:
        df["distance_km"] = 0.0

    # Moving time: seconds -> hours
    if "moving_time" in df.columns:
        df["moving_hours"] = df["moving_time"].fillna(0) / 3600.0
    else:
        df["moving_hours"] = 0.0

    # Ensure sport_type
    if "sport_type" not in df.columns:
        df["sport_type"] = df.get("type", "Unknown")
    else:
        df["sport_type"] = df["sport_type"].fillna(df.get("type", "Unknown"))

    # Optional: average speed in km/h if available
    if "average_speed" in df.columns:
        df["avg_speed_kmh"] = df["average_speed"] * 3.6
    else:
        df["avg_speed_kmh"] = np.nan

    return df


# ============================================================
# METRICS & HELPERS
# ============================================================

def compute_year_summary(df: pd.DataFrame, year: int) -> pd.Series:
    year_df = df[df["year"] == year]
    if year_df.empty:
        return pd.Series(
            {
                "year": year,
                "total_km": 0.0,
                "total_hours": 0.0,
                "n_activities": 0,
                "n_active_days": 0,
            }
        )

    return pd.Series(
        {
            "year": year,
            "total_km": year_df["distance_km"].sum(),
            "total_hours": year_df["moving_hours"].sum(),
            "n_activities": len(year_df),
            "n_active_days": year_df["date"].nunique(),
        }
    )


def compute_training_streaks(df: pd.DataFrame, year: int | None = None) -> dict:
    from datetime import timedelta

    tmp = df.copy()
    if year is not None:
        tmp = tmp[tmp["year"] == year]

    if tmp.empty:
        return {
            "longest_streak": 0,
            "longest_start": None,
            "longest_end": None,
            "current_streak": 0,
        }

    days = sorted(tmp["date"].dropna().unique())
    if not days:
        return {
            "longest_streak": 0,
            "longest_start": None,
            "longest_end": None,
            "current_streak": 0,
        }

    longest = current = 1
    longest_start = current_start = days[0]
    longest_end = days[0]

    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days == 1:
            current += 1
        else:
            if current > longest:
                longest = current
                longest_start = current_start
                longest_end = days[i - 1]
            current = 1
            current_start = days[i]

    if current > longest:
        longest = current
        longest_start = current_start
        longest_end = days[-1]

    # current streak -> if last activity was today or yesterday etc
    today = date.today()
    if (today - days[-1]).days == 0:
        current_streak = current
    else:
        current_streak = 0

    return {
        "longest_streak": int(longest),
        "longest_start": longest_start,
        "longest_end": longest_end,
        "current_streak": int(current_streak),
    }


def best_and_worst_periods(df: pd.DataFrame, year: int) -> dict:
    year_df = df[df["year"] == year].copy()
    if year_df.empty:
        return {}

    # Monthly
    by_month = year_df.groupby("month")["distance_km"].sum()
    best_month = int(by_month.idxmax())
    worst_month = int(by_month.idxmin())

    # Weekly
    tmp = year_df.set_index("start_date_local")
    weekly = tmp["distance_km"].resample("W-MON").sum()
    best_week_start = weekly.idxmax().date()
    worst_week_start = weekly.idxmin().date()

    return {
        "best_month": best_month,
        "best_month_km": float(by_month.max()),
        "worst_month": worst_month,
        "worst_month_km": float(by_month.min()),
        "best_week_start": best_week_start,
        "best_week_km": float(weekly.max()),
        "worst_week_start": worst_week_start,
        "worst_week_km": float(weekly.min()),
    }


def forecast_year_distance(df: pd.DataFrame, year: int) -> dict:
    today = date.today()
    year_df = df[df["year"] == year].copy()
    if year_df.empty:
        return {}

    start = date(year, 1, 1)
    end = min(today, date(year, 12, 31))
    days_passed = (end - start).days + 1

    total_days = 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365
    total_km = year_df["distance_km"].sum()
    daily_avg = total_km / days_passed
    forecast_total = daily_avg * total_days

    return {
        "current_total_km": total_km,
        "daily_avg_km": daily_avg,
        "forecast_total_km": forecast_total,
        "days_passed": days_passed,
        "total_days": total_days,
    }


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Strava Dashboard",
        page_icon="🚴",
        layout="wide",
    )

    st.title("🚴 Strava Yearly Dashboard")

    # Load data
    df = load_activities(DATA_PATH)

    # Sidebar controls
    years = sorted(df["year"].dropna().unique())
    default_year = years[-1] if years else None
    year = st.sidebar.selectbox("Select year", years, index=len(years) - 1)

    sport_types = sorted(df["sport_type"].dropna().unique())
    selected_sports = st.sidebar.multiselect(
        "Filter by sport type (optional)", ["All"] + sport_types, default=["All"]
    )

    filtered_df = df[df["year"] == year]
    if "All" not in selected_sports:
        filtered_df = filtered_df[filtered_df["sport_type"].isin(selected_sports)]

    # Goals (optional)
    st.sidebar.markdown("### 🎯 Goals")
    goal_km = st.sidebar.number_input("Yearly distance (km)", value=6000.0, step=100.0)
    goal_hours = st.sidebar.number_input("Yearly hours", value=350.0, step=10.0)
    goal_active_days = st.sidebar.number_input("Active days", value=220, step=5)

    # Summary metrics
    summary = compute_year_summary(filtered_df, year)
    streaks = compute_training_streaks(filtered_df, year)
    bw = best_and_worst_periods(filtered_df, year)
    fc = forecast_year_distance(filtered_df, year)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total distance (km)", f"{summary['total_km']:.0f}")
    col2.metric("Total hours", f"{summary['total_hours']:.1f}")
    col3.metric("Activities", int(summary["n_activities"]))
    col4.metric("Active days", int(summary["n_active_days"]))

    # Goal progress
    if goal_km > 0 or goal_hours > 0 or goal_active_days > 0:
        st.subheader("🎯 Goal Progress")

        gcol1, gcol2, gcol3 = st.columns(3)
        if goal_km > 0:
            pct_km = summary["total_km"] / goal_km * 100
            gcol1.metric("Distance goal", f"{pct_km:.0f}% of {goal_km:.0f} km")
        if goal_hours > 0:
            pct_hours = summary["total_hours"] / goal_hours * 100
            gcol2.metric("Hours goal", f"{pct_hours:.0f}% of {goal_hours:.0f} h")
        if goal_active_days > 0:
            pct_days = summary["n_active_days"] / goal_active_days * 100
            gcol3.metric("Active days goal", f"{pct_days:.0f}% of {goal_active_days}")

    # Streaks & best/worst
    st.subheader("📅 Streaks & Highlights")

    scol1, scol2, scol3 = st.columns(3)
    scol1.write(
        f"**Longest streak:** {streaks['longest_streak']} days "
        f"({streaks['longest_start']} → {streaks['longest_end']})"
    )
    scol2.write(f"**Current streak:** {streaks['current_streak']} days")
    if bw:
        scol3.write(
            f"**Best month:** {bw['best_month']} ({bw['best_month_km']:.0f} km)  \n"
            f"**Best week start:** {bw['best_week_start']} "
            f"({bw['best_week_km']:.0f} km)"
        )

    st.markdown("---")

    # Tabs for charts
    tab1, tab2, tab3, tab4 = st.tabs(
        ["By Sport", "By Month", "Weekly Trend", "Raw Table"]
    )

    with tab1:
        st.subheader("Distance by Sport")
        by_sport = (
            filtered_df.groupby("sport_type")
            .agg(
                total_km=("distance_km", "sum"),
                total_hours=("moving_hours", "sum"),
                n_activities=("distance_km", "count"),
            )
            .reset_index()
        )
        if not by_sport.empty:
            fig = px.bar(
                by_sport,
                x="sport_type",
                y="total_km",
                hover_data=["total_hours", "n_activities"],
                labels={"total_km": "Distance (km)", "sport_type": "Sport"},
                title=f"Distance per Sport Type – {year}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activities for this selection.")

    with tab2:
        st.subheader("Distance by Month")
        month_df = (
            filtered_df.groupby("month")["distance_km"].sum().reindex(range(1, 13), fill_value=0).reset_index()
        )
        month_df["month_name"] = month_df["month"].apply(
            lambda m: date(2000, m, 1).strftime("%b")
        )
        fig = px.bar(
            month_df,
            x="month_name",
            y="distance_km",
            labels={"distance_km": "Distance (km)", "month_name": "Month"},
            title=f"Distance per Month – {year}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Weekly Distance Trend")
        if not filtered_df.empty:
            tmp = filtered_df.set_index("start_date_local")
            weekly = tmp["distance_km"].resample("W-MON").sum().reset_index()
            fig = px.line(
                weekly,
                x="start_date_local",
                y="distance_km",
                labels={"distance_km": "Distance (km)", "start_date_local": "Week"},
                title=f"Weekly Distance – {year}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for weekly chart.")

    with tab4:
        st.subheader("Raw Activities (filtered)")
        st.dataframe(
            filtered_df[
                [
                    "date",
                    "sport_type",
                    "name",
                    "distance_km",
                    "moving_hours",
                    "avg_speed_kmh",
                ]
            ].sort_values("date", ascending=False),
            use_container_width=True,
            height=400,
        )


if __name__ == "__main__":
    main()
