from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from endurance_lab.config import ftp_at, load_athlete_config
from endurance_lab.db import connect, init_db


METRIC_VERSION = 1
POWER_DURATIONS = (5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600)


@dataclass(frozen=True)
class DecouplingResult:
    value_pct: float | None
    status: str
    reason: str | None
    first_output: float | None = None
    second_output: float | None = None
    first_hr: float | None = None
    second_hr: float | None = None
    hr_coverage: float | None = None
    moving_ratio: float | None = None


def normalized_power(power: Sequence[float | None], window_seconds: int = 30) -> float | None:
    clean_count = sum(value is not None for value in power)
    if len(power) < window_seconds or clean_count / max(len(power), 1) < 0.7:
        return None
    filled = _short_gap_fill(power, max_gap=5)
    rolling: list[float] = []
    window_sum = 0.0
    missing = 0
    for index, value in enumerate(filled):
        if value is None:
            missing += 1
        else:
            window_sum += float(value)
        if index >= window_seconds:
            old = filled[index - window_seconds]
            if old is None:
                missing -= 1
            else:
                window_sum -= float(old)
        if index >= window_seconds - 1 and missing == 0:
            rolling.append(window_sum / window_seconds)
    return (sum(value**4 for value in rolling) / len(rolling)) ** 0.25 if rolling else None


def best_power_durations(
    power: Sequence[float | None], durations: Sequence[int] = POWER_DURATIONS
) -> dict[int, float]:
    filled = _short_gap_fill(power, max_gap=2)
    results: dict[int, float] = {}
    for duration in durations:
        if duration <= 0 or len(filled) < duration:
            continue
        best: float | None = None
        total = 0.0
        missing = 0
        for index, value in enumerate(filled):
            if value is None:
                missing += 1
            else:
                total += float(value)
            if index >= duration:
                old = filled[index - duration]
                if old is None:
                    missing -= 1
                else:
                    total -= float(old)
            if index >= duration - 1 and missing / duration <= 0.05:
                candidate = total / (duration - missing)
                best = candidate if best is None else max(best, candidate)
        if best is not None:
            results[int(duration)] = best
    return results


def zone_seconds(
    values: Sequence[float | None],
    boundaries: Sequence[float],
    elapsed: Sequence[float] | None = None,
    moving: Sequence[bool | None] | None = None,
) -> list[float]:
    if len(boundaries) < 2:
        return []
    totals = [0.0 for _ in range(len(boundaries) - 1)]
    for index in range(max(0, len(values) - 1)):
        value = values[index]
        if value is None or (moving is not None and moving[index] is False):
            continue
        seconds = 1.0 if elapsed is None else max(0.0, min(10.0, elapsed[index + 1] - elapsed[index]))
        for zone in range(len(boundaries) - 1):
            if boundaries[zone] <= value < boundaries[zone + 1]:
                totals[zone] += seconds
                break
    return totals


def first_second_halves(
    elapsed: Sequence[float],
    output: Sequence[float | None],
    heart_rate: Sequence[float | None],
    moving: Sequence[bool | None] | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    if not elapsed:
        return None, None, None, None
    midpoint = (elapsed[0] + elapsed[-1]) / 2
    first_indexes = [
        index
        for index, value in enumerate(elapsed)
        if value <= midpoint and (moving is None or moving[index] is not False)
    ]
    second_indexes = [
        index
        for index, value in enumerate(elapsed)
        if value > midpoint and (moving is None or moving[index] is not False)
    ]
    return (
        _mean(output[index] for index in first_indexes),
        _mean(output[index] for index in second_indexes),
        _mean(heart_rate[index] for index in first_indexes),
        _mean(heart_rate[index] for index in second_indexes),
    )


def aerobic_decoupling(
    elapsed: Sequence[float],
    output: Sequence[float | None],
    heart_rate: Sequence[float | None],
    moving: Sequence[bool | None] | None = None,
    *,
    min_duration_seconds: float = 3600,
    min_hr_coverage: float = 0.8,
    min_moving_ratio: float = 0.85,
    max_output_cv: float = 0.35,
) -> DecouplingResult:
    if len(elapsed) < 2 or elapsed[-1] - elapsed[0] < min_duration_seconds:
        return DecouplingResult(None, "not_applicable", "activity_too_short")
    moving_mask = [True] * len(elapsed) if moving is None else [value is not False for value in moving]
    moving_count = sum(moving_mask)
    moving_ratio = moving_count / len(moving_mask)
    if moving_ratio < min_moving_ratio:
        return DecouplingResult(None, "not_applicable", "excessive_stops", moving_ratio=moving_ratio)
    eligible = [index for index, is_moving in enumerate(moving_mask) if is_moving and output[index] is not None]
    if not eligible:
        return DecouplingResult(None, "not_applicable", "missing_output", moving_ratio=moving_ratio)
    coverage = sum(heart_rate[index] is not None for index in eligible) / len(eligible)
    if coverage < min_hr_coverage:
        return DecouplingResult(
            None, "not_applicable", "insufficient_hr", hr_coverage=coverage, moving_ratio=moving_ratio
        )
    output_values = [float(output[index]) for index in eligible if output[index] is not None]
    output_mean = statistics.fmean(output_values)
    cv = statistics.pstdev(output_values) / output_mean if output_mean > 0 else math.inf
    if cv > max_output_cv:
        return DecouplingResult(
            None, "not_applicable", "variable_or_interval_session", hr_coverage=coverage, moving_ratio=moving_ratio
        )
    first_output, second_output, first_hr, second_hr = first_second_halves(
        elapsed, output, heart_rate, moving
    )
    if not all(value is not None and value > 0 for value in (first_output, second_output, first_hr, second_hr)):
        return DecouplingResult(
            None, "not_applicable", "insufficient_half_data", hr_coverage=coverage, moving_ratio=moving_ratio
        )
    first_efficiency = float(first_output) / float(first_hr)
    second_efficiency = float(second_output) / float(second_hr)
    value = (first_efficiency - second_efficiency) / first_efficiency * 100
    return DecouplingResult(
        value,
        "applicable",
        None,
        first_output,
        second_output,
        first_hr,
        second_hr,
        coverage,
        moving_ratio,
    )


def rolling_load(
    daily_loads: dict[date, float], fitness_days: int = 42, fatigue_days: int = 7
) -> list[dict[str, float | date]]:
    if not daily_loads:
        return []
    current = min(daily_loads)
    end = max(daily_loads)
    fitness = fatigue = 0.0
    rows: list[dict[str, float | date]] = []
    while current <= end:
        load = float(daily_loads.get(current, 0.0))
        fitness += (load - fitness) / fitness_days
        fatigue += (load - fatigue) / fatigue_days
        rows.append(
            {"day": current, "load": load, "fitness": fitness, "fatigue": fatigue, "form": fitness - fatigue}
        )
        current += timedelta(days=1)
    return rows


def analyze_database(
    database: str | Path | None = None,
    config_path: str | Path | None = None,
    activity_ids: Iterable[int] | None = None,
) -> int:
    init_db(database)
    config = load_athlete_config(config_path)
    computed_at = datetime.now(timezone.utc).isoformat()
    selected_ids = sorted({int(value) for value in activity_ids}) if activity_ids is not None else None
    with connect(database) as connection:
        if selected_ids is None:
            activities = connection.execute("SELECT * FROM activities ORDER BY started_at").fetchall()
        elif selected_ids:
            placeholders = ",".join("?" for _ in selected_ids)
            activities = connection.execute(
                f"SELECT * FROM activities WHERE id IN ({placeholders}) ORDER BY started_at",
                selected_ids,
            ).fetchall()
        else:
            return 0
        connection.execute("BEGIN IMMEDIATE")
        try:
            if selected_ids is None:
                connection.execute("DELETE FROM power_curve_results")
                connection.execute("DELETE FROM derived_activity_metrics")
            else:
                placeholders = ",".join("?" for _ in selected_ids)
                connection.execute(f"DELETE FROM power_curve_results WHERE activity_id IN ({placeholders})", selected_ids)
                connection.execute(f"DELETE FROM derived_activity_metrics WHERE activity_id IN ({placeholders})", selected_ids)
            connection.execute("DELETE FROM daily_training_load")
            for activity in activities:
                streams = connection.execute(
                    "SELECT * FROM activity_streams WHERE activity_id = ? ORDER BY sequence",
                    (activity["id"],),
                ).fetchall()
                metrics, best_power = derive_activity(dict(activity), [dict(row) for row in streams], config)
                _write_metrics(connection, int(activity["id"]), metrics, computed_at)
                connection.executemany(
                    """INSERT INTO power_curve_results
                       (activity_id, duration_seconds, best_power_w, computed_at)
                       VALUES (?, ?, ?, ?)""",
                    [(activity["id"], duration, watts, computed_at) for duration, watts in best_power.items()],
                )
            # Daily fitness/fatigue depends on chronological history, but it can be
            # rebuilt cheaply from persisted per-activity metrics.  Raw streams and
            # expensive sport analytics are recalculated only for affected IDs.
            load_rows = connection.execute(
                """SELECT a.*, d.selected_load
                   FROM activities a
                   LEFT JOIN derived_activity_metrics d ON d.activity_id=a.id
                   ORDER BY a.started_at"""
            ).fetchall()
            _write_daily_load(connection, [dict(row) for row in load_rows], config, computed_at)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return len(activities)


def derive_activity(
    activity: dict[str, Any], streams: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, float]]:
    elapsed, power, hr, speed, moving = _resampled_streams(streams)
    sport = str(activity.get("sport") or "other")
    output = power if sport == "cycling" else speed
    analysis_cfg = config.get("analysis", {})
    decoupling = aerobic_decoupling(
        elapsed,
        output,
        hr,
        moving,
        min_duration_seconds=float(analysis_cfg.get("decoupling_min_duration_minutes", 60)) * 60,
        min_hr_coverage=float(analysis_cfg.get("decoupling_min_hr_coverage", 0.8)),
        min_moving_ratio=float(analysis_cfg.get("decoupling_min_moving_ratio", 0.85)),
        max_output_cv=float(analysis_cfg.get("decoupling_max_output_cv", 0.35)),
    )
    np_value = normalized_power(power) if sport == "cycling" else None
    ftp = ftp_at(config, activity.get("started_at")) if sport == "cycling" else None
    intensity = np_value / ftp if np_value is not None and ftp else None
    duration = float(activity.get("moving_seconds") or activity.get("elapsed_seconds") or 0)
    avg_power = _mean(power)
    tss = None
    if np_value is not None and avg_power is not None and ftp and duration > 0:
        tss = duration * np_value * (np_value / ftp) / (ftp * 3600) * 100

    hr_boundaries = [float(value) for value in config.get("heart_rate", {}).get("zones_bpm", [])]
    hr_zones = zone_seconds(hr, hr_boundaries, elapsed, moving) if hr_boundaries else []
    hr_load = sum(seconds / 60 * (index + 1) for index, seconds in enumerate(hr_zones)) if hr_zones else None

    power_zones: list[float] = []
    if ftp:
        percentages = config.get("cycling", {}).get("power_zone_percentages", [])
        boundaries = [float(value) * ftp for value in percentages]
        power_zones = zone_seconds(power, boundaries, elapsed, moving)

    load, method = _select_load(sport, duration, tss, hr_load)
    output_avg = _mean(value for value, is_moving in zip(output, moving) if is_moving is not False)
    hr_avg = _mean(value for value, is_moving in zip(hr, moving) if is_moving is not False)
    efficiency = output_avg / hr_avg if output_avg and hr_avg else None
    first_out, second_out, first_hr, second_hr = first_second_halves(elapsed, output, hr, moving)
    late_fade = (first_out - second_out) / first_out * 100 if first_out and second_out else None
    distance_m = float(activity.get("distance_m") or 0)
    pace = duration / (distance_m / 1000) if sport == "running" and distance_m > 0 else None
    best = best_power_durations(power) if sport == "cycling" else {}
    return (
        {
            "ftp_w": ftp,
            "normalized_power_w": np_value,
            "intensity_factor": intensity,
            "estimated_tss": tss,
            "estimated_hr_load": hr_load,
            "selected_load": load,
            "load_method": method,
            "efficiency_factor": efficiency,
            "aerobic_decoupling_pct": decoupling.value_pct,
            "decoupling_status": decoupling.status,
            "decoupling_reason": decoupling.reason,
            "first_half_output": first_out,
            "second_half_output": second_out,
            "first_half_hr": first_hr,
            "second_half_hr": second_hr,
            "late_fade_pct": late_fade,
            "pace_seconds_per_km": pace,
            "hr_coverage": decoupling.hr_coverage,
            "moving_ratio": decoupling.moving_ratio,
            "hr_zones_json": json.dumps(hr_zones),
            "power_zones_json": json.dumps(power_zones),
            "best_power_json": json.dumps(best),
        },
        best,
    )


def _resampled_streams(
    streams: list[dict[str, Any]],
) -> tuple[list[float], list[float | None], list[float | None], list[float | None], list[bool | None]]:
    usable = [row for row in streams if row.get("elapsed_seconds") is not None]
    if not usable:
        return [], [], [], [], []
    by_second: dict[int, dict[str, Any]] = {int(round(float(row["elapsed_seconds"]))): row for row in usable}
    start, end = min(by_second), max(by_second)
    elapsed: list[float] = []
    power: list[float | None] = []
    hr: list[float | None] = []
    speed: list[float | None] = []
    moving: list[bool | None] = []
    previous: dict[str, Any] | None = None
    previous_second = start
    for second in range(start, end + 1):
        row = by_second.get(second)
        if row is not None:
            previous = row
            previous_second = second
        use = previous if previous is not None and second - previous_second <= 5 else None
        elapsed.append(float(second))
        power.append(_float(use.get("power_w")) if use else None)
        hr.append(_float(use.get("heart_rate")) if use else None)
        speed.append(_float(use.get("speed_mps")) if use else None)
        moving.append(None if use is None or use.get("moving") is None else bool(use.get("moving")))
    return elapsed, power, hr, speed, moving


def _select_load(
    sport: str, duration_seconds: float, tss: float | None, hr_load: float | None
) -> tuple[float, str]:
    if tss is not None:
        return tss, "estimated_power_tss"
    if hr_load is not None and hr_load > 0:
        return hr_load, "estimated_edwards_hr"
    minutes = duration_seconds / 60
    factor = {"running": 1.0, "cycling": 0.65, "swimming": 0.8, "strength": 0.5}.get(sport, 0.5)
    return minutes * factor, "duration_only_estimate"


def _write_metrics(connection, activity_id: int, metrics: dict[str, Any], computed_at: str) -> None:
    columns = [
        "ftp_w", "normalized_power_w", "intensity_factor", "estimated_tss",
        "estimated_hr_load", "selected_load", "load_method", "efficiency_factor",
        "aerobic_decoupling_pct", "decoupling_status", "decoupling_reason",
        "first_half_output", "second_half_output", "first_half_hr", "second_half_hr",
        "late_fade_pct", "pace_seconds_per_km", "hr_coverage", "moving_ratio",
        "hr_zones_json", "power_zones_json", "best_power_json",
    ]
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO derived_activity_metrics (activity_id, metric_version, computed_at, {', '.join(columns)}) VALUES (?, ?, ?, {placeholders})",
        (activity_id, METRIC_VERSION, computed_at, *(metrics[column] for column in columns)),
    )


def _write_daily_load(connection, rows: list[dict[str, Any]], config: dict[str, Any], computed_at: str) -> None:
    if not rows:
        return
    grouped: dict[date, dict[str, Any]] = {}
    long_limits = config.get("load_model", {}).get("long_session_minutes", {})
    hard_limit = float(config.get("load_model", {}).get("hard_session_load", 100))
    for row in rows:
        day = datetime.fromisoformat(str(row["started_at"])).date()
        bucket = grouped.setdefault(day, {"total": 0.0, "duration": 0.0, "sessions": 0, "hard": 0, "long": 0})
        load = float(row.get("selected_load") or 0)
        sport = str(row.get("sport") or "other")
        bucket["total"] += load
        bucket[sport if sport in {"cycling", "running", "swimming", "strength"} else "other"] = bucket.get(sport, 0.0) + load
        duration = float(row.get("moving_seconds") or row.get("elapsed_seconds") or 0)
        bucket["duration"] += duration
        bucket["sessions"] += 1
        bucket["hard"] += int(load >= hard_limit)
        bucket["long"] += int(duration / 60 >= float(long_limits.get(sport, math.inf)))

    model = config.get("load_model", {})
    load_series = {day: values["total"] for day, values in grouped.items()}
    rolling = rolling_load(
        load_series,
        int(model.get("fitness_time_constant_days", 42)),
        int(model.get("fatigue_time_constant_days", 7)),
    )
    for item in rolling:
        day = item["day"]
        bucket = grouped.get(day, {})
        connection.execute(
            """INSERT INTO daily_training_load (
                day, total_load, cycling_load, running_load, swimming_load,
                strength_load, other_load, duration_seconds, sessions,
                hard_sessions, long_sessions, fitness, fatigue, form, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                day.isoformat(), item["load"], bucket.get("cycling", 0.0),
                bucket.get("running", 0.0), bucket.get("swimming", 0.0),
                bucket.get("strength", 0.0), bucket.get("other", 0.0),
                bucket.get("duration", 0.0), bucket.get("sessions", 0),
                bucket.get("hard", 0), bucket.get("long", 0),
                item["fitness"], item["fatigue"], item["form"], computed_at,
            ),
        )


def _short_gap_fill(values: Sequence[float | None], max_gap: int) -> list[float | None]:
    result = list(values)
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        if index - start <= max_gap and start > 0 and index < len(result):
            before, after = float(result[start - 1]), float(result[index])
            span = index - start + 1
            for offset in range(index - start):
                result[start + offset] = before + (after - before) * (offset + 1) / span
    return result


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _float(value: Any) -> float | None:
    try:
        result = float(value) if value is not None else None
        return result if result is not None and math.isfinite(result) else None
    except (TypeError, ValueError):
        return None
