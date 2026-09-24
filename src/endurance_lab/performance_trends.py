from __future__ import annotations

import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from endurance_lab.coaching_models import TrendSignal
from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db


def performance_trends(
    as_of: date,
    database: str | Path | None = None,
) -> dict[str, TrendSignal]:
    """Return trends using only activities strictly before ``as_of``."""
    init_db(database)
    settings = load_athlete_config().get("coaching", {}).get("trends", {})
    lookback = int(settings.get("lookback_days", 84))
    window = int(settings.get("comparison_window_days", 42))
    minimum = int(settings.get("minimum_sessions", 4))
    meaningful = float(settings.get("meaningful_change_fraction", 0.05))
    start = as_of - timedelta(days=lookback)
    split = as_of - timedelta(days=window)
    with connect(database) as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT a.id, substr(a.started_at, 1, 10) AS day, a.sport,
                      COALESCE(a.moving_seconds, a.elapsed_seconds, 0) AS duration_seconds,
                      a.avg_hr, a.distance_m, d.normalized_power_w, d.efficiency_factor,
                      d.aerobic_decoupling_pct, d.late_fade_pct, d.pace_seconds_per_km,
                      p.best_power_w AS power_20m
               FROM activities a
               LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
               LEFT JOIN power_curve_results p ON p.activity_id = a.id AND p.duration_seconds = 1200
               WHERE a.started_at >= ? AND a.started_at < ?
               ORDER BY a.started_at""",
            (start.isoformat(), as_of.isoformat()),
        )]
    return {
        sport: _sport_trend(sport, rows, split, minimum, meaningful, start, as_of)
        for sport in ("cycling", "running", "swimming")
    }


def _sport_trend(
    sport: str,
    rows: list[dict[str, Any]],
    split: date,
    minimum: int,
    meaningful: float,
    start: date,
    as_of: date,
) -> TrendSignal:
    relevant = [row for row in rows if row["sport"] == sport and row["duration_seconds"] >= 1200]
    metric_name, higher_is_better, values = _metric(sport, relevant)
    earlier = [value for day, value in values if date.fromisoformat(day) < split]
    recent = [value for day, value in values if date.fromisoformat(day) >= split]
    window_text = f"{start.isoformat()} to {(as_of - timedelta(days=1)).isoformat()}"
    if len(values) < minimum or len(earlier) < 2 or len(recent) < 2:
        return TrendSignal(
            sport, "UNCERTAIN", "low",
            (
                f"{len(values)} comparable {metric_name} observations "
                f"({len(earlier)} earlier, {len(recent)} recent); at least two per window required",
            ),
            window_text,
        )
    old = statistics.median(earlier)
    new = statistics.median(recent)
    if old == 0:
        return TrendSignal(sport, "UNCERTAIN", "low", (f"{metric_name} baseline is unavailable",), window_text)
    raw_change = (new - old) / abs(old)
    improvement = raw_change if higher_is_better else -raw_change
    state = "IMPROVING" if improvement >= meaningful else "DECLINING" if improvement <= -meaningful else "STABLE"
    confidence = "high" if len(values) >= max(8, minimum * 2) else "medium"
    direction = "higher" if raw_change > 0 else "lower"
    evidence = (
        f"median {metric_name} changed from {old:.2f} to {new:.2f} ({abs(raw_change):.0%} {direction})",
        f"comparison uses {len(earlier)} earlier and {len(recent)} recent comparable sessions",
    )
    return TrendSignal(sport, state, confidence, evidence, window_text)


def _metric(sport: str, rows: list[dict[str, Any]]) -> tuple[str, bool, list[tuple[str, float]]]:
    if sport == "cycling":
        efficiency, band = _comparable_values(rows, "efficiency_factor")
        if len(efficiency) >= 4:
            return f"power/HR efficiency ({band})", True, efficiency
        power, band = _comparable_values(rows, "power_20m")
        return f"20-minute power ({band})", True, power
    if sport == "running":
        efficiency, band = _comparable_values(rows, "efficiency_factor")
        if len(efficiency) >= 4:
            return f"pace/HR efficiency ({band})", True, efficiency
        pace, band = _comparable_values(rows, "pace_seconds_per_km")
        return f"pace per kilometre ({band})", False, pace
    swim_rows = [dict(row, swim_pace=float(row["duration_seconds"]) / (float(row["distance_m"]) / 100))
                 for row in rows if row["distance_m"] and row["duration_seconds"]]
    pace, band = _comparable_values(swim_rows, "swim_pace")
    return f"pace per 100 m ({band})", False, pace


def _comparable_values(rows: list[dict[str, Any]], field: str) -> tuple[list[tuple[str, float]], str]:
    buckets: dict[str, list[tuple[str, float]]] = {
        "20-59 min": [], "60-119 min": [], "120+ min": [],
    }
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        minutes = float(row["duration_seconds"]) / 60
        band = "20-59 min" if minutes < 60 else "60-119 min" if minutes < 120 else "120+ min"
        buckets[band].append((row["day"], float(value)))
    band, values = max(buckets.items(), key=lambda item: len(item[1]))
    return values, band
