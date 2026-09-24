from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from endurance_lab.coaching_models import SessionCost
from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db


MODEL_VERSION = 1
LEVELS = {"low": 0, "moderate": 1, "high": 2}


def session_cost(
    activity_id: int | str,
    database: str | Path | None = None,
    persist: bool = True,
) -> SessionCost:
    init_db(database)
    config = load_athlete_config()
    with connect(database) as connection:
        row = connection.execute(
            _COST_QUERY + " WHERE a.id = ? OR a.source_activity_id = ? ORDER BY (a.id = ?) DESC LIMIT 1",
            (activity_id, str(activity_id), activity_id),
        ).fetchone()
        if not row:
            raise ValueError(f"Activity {activity_id} was not found")
        record = dict(row)
        strength = _strength_context(connection, record, config)
        result = assess_session_cost(record, config, strength)
        if persist:
            _persist(connection, result)
            connection.commit()
    return result


def classify_session_costs(
    database: str | Path | None = None,
    before: str | None = None,
    persist: bool = True,
) -> list[SessionCost]:
    init_db(database)
    config = load_athlete_config()
    with connect(database) as connection:
        query = _COST_QUERY
        params: tuple[Any, ...] = ()
        if before:
            query += " WHERE a.started_at < ?"
            params = (before,)
        query += " ORDER BY a.started_at, a.id"
        results = []
        for row in connection.execute(query, params):
            record = dict(row)
            results.append(assess_session_cost(record, config, _strength_context(connection, record, config)))
        if persist:
            for result in results:
                _persist(connection, result)
            connection.commit()
    return results


def assess_session_cost(
    row: dict[str, Any],
    config: dict[str, Any] | None = None,
    strength_context: dict[str, Any] | None = None,
) -> SessionCost:
    config = config or load_athlete_config()
    sport = str(row.get("sport") or "other").lower()
    if sport == "strength":
        return _strength_cost(row, config, strength_context or {})
    return _endurance_cost(row, config)


def _endurance_cost(row: dict[str, Any], config: dict[str, Any]) -> SessionCost:
    sport = str(row.get("sport") or "other").lower()
    duration = _number(row.get("moving_seconds")) or _number(row.get("elapsed_seconds")) or 0.0
    minutes = duration / 60
    settings = config.get("coaching", {}).get("cost_model", {})
    duration_settings = settings.get("endurance_duration_minutes", {})
    low_minutes = float(duration_settings.get("low", 45))
    high_minutes = float(duration_settings.get("high", 150))
    duration_cost = "low" if minutes < low_minutes else "high" if minutes >= high_minutes else "moderate"
    intensity_factor = _number(row.get("intensity_factor"))
    selected_load = _number(row.get("selected_load"))
    hard_load = float(config.get("load_model", {}).get("hard_session_load", 100))
    ftp = _number(row.get("ftp_w"))
    avg_power = _number(row.get("avg_power_w"))
    avg_hr = _number(row.get("avg_hr"))
    aerobic_hr = _number(config.get("athlete", {}).get("aerobic_hr_upper_bpm"))
    hr_high = _upper_zone_fraction(row.get("hr_zones_json"))
    power_high = _upper_zone_fraction(row.get("power_zones_json"))
    text = " ".join(str(row.get(key) or "") for key in (
        "name", "planned_title", "planned_type", "planned_intensity", "planned_description"
    )).lower()
    evidence: list[str] = [f"{minutes:.0f} min measured duration"]
    context_flags: list[str] = []
    temp = max(_number(row.get("max_temperature_c")) or -math.inf, _number(row.get("avg_temperature_c")) or -math.inf)
    if temp >= float(settings.get("heat_threshold_c", 30)):
        context_flags.append("heat")
        evidence.append(f"temperature reached {temp:.0f}°C; HR-based performance inference is less certain")
    if any(re.search(rf"\b{re.escape(token)}\b", text) for token in ("mechanical", "puncture", "traffic", "route", "interruption")):
        context_flags.append("route_interruption")

    explicit_quality = any(token in text for token in ("threshold", "treshold", "vo2", "interval", "race"))
    recovery_signal = False
    if sport == "cycling":
        very_low_power = avg_power is not None and ftp is not None and avg_power / ftp < 0.45
        low_hr = avg_hr is not None and aerobic_hr is not None and avg_hr < aerobic_hr * 0.72
        recovery_signal = (very_low_power and low_hr) or (
            very_low_power and max(hr_high, power_high) < 0.05
        )
        if very_low_power:
            evidence.append(f"average power was {avg_power / ftp:.0%} of configured FTP")
        if low_hr:
            evidence.append(f"average HR {avg_hr:.0f} bpm was well below the aerobic ceiling")
    elif sport == "running":
        low_hr = avg_hr is not None and aerobic_hr is not None and avg_hr < aerobic_hr * 0.88
        recovery_signal = low_hr and hr_high < 0.05 and not explicit_quality
        if low_hr:
            evidence.append("heart rate remained compatible with easy running")
    elif sport == "swimming":
        recovery_signal = not explicit_quality and hr_high < 0.05 and minutes < 45
    elif sport == "other":
        recovery_signal = minutes < 60 and hr_high < 0.05

    high_signal = explicit_quality or (intensity_factor or 0) >= 0.88 or max(hr_high, power_high) >= 0.20 or (selected_load or 0) >= hard_load
    moderate_signal = (intensity_factor or 0) >= 0.68 or max(hr_high, power_high) >= 0.08 or (selected_load or 0) >= hard_load * 0.60
    if high_signal:
        intensity_cost = cardiovascular = "high"
        evidence.append("quality-session or high-intensity evidence is present")
    elif recovery_signal:
        intensity_cost = cardiovascular = "low"
        evidence.append("measured output and cardiovascular response indicate recovery-level work")
    elif moderate_signal:
        intensity_cost = cardiovascular = "moderate"
        evidence.append("measured intensity indicates moderate aerobic cost")
    else:
        intensity_cost = cardiovascular = "low" if minutes < 45 else "moderate"
        evidence.append("no high-intensity signal is present")

    muscular = "high" if sport == "running" and high_signal else "moderate" if (
        sport == "running" or high_signal
    ) else "low"
    muscle_load = "lower" if sport in {"cycling", "running"} else "full" if sport == "swimming" else "none"
    systemic = _max_level(cardiovascular, muscular)
    if recovery_signal:
        systemic = "low"
        cost_class = "recovery" if intensity_cost == "low" else "aerobic"
    elif high_signal:
        cost_class = "high"
    elif systemic == "moderate" or duration_cost == "high":
        cost_class = "moderate" if cardiovascular == "moderate" else "aerobic"
    else:
        cost_class = "aerobic"
    confidence_points = sum(value is not None for value in (intensity_factor, avg_hr, avg_power))
    confidence = "high" if confidence_points >= 2 or explicit_quality else "medium" if confidence_points else "low"
    if context_flags and confidence == "high":
        confidence = "medium"
    return SessionCost(
        int(row["id"]), cost_class, systemic, cardiovascular, muscular, muscle_load,
        sport if sport in {"cycling", "running", "swimming"} else "mixed",
        intensity_cost, duration_cost, confidence, tuple(evidence), tuple(dict.fromkeys(context_flags)), {},
    )


def _strength_cost(row: dict[str, Any], config: dict[str, Any], context: dict[str, Any]) -> SessionCost:
    duration = (_number(row.get("moving_seconds")) or _number(row.get("elapsed_seconds")) or 0) / 60
    counts = Counter(context.get("movement_counts", {}))
    working = Counter(context.get("working_sets", {}))
    lower = int(working.get("lower", 0))
    upper = int(working.get("upper", 0))
    full = int(working.get("full", 0))
    settings = config.get("coaching", {}).get("cost_model", {}).get("strength", {})
    meaningful = int(settings.get("meaningful_working_sets", 3))
    high_sets = int(settings.get("high_working_sets", 5))
    historical_high = bool(context.get("near_recent_high"))
    if lower >= high_sets or (lower >= meaningful and historical_high):
        muscular, muscle_load = "high", "lower"
    elif lower >= meaningful:
        muscular, muscle_load = "moderate", "lower"
    elif full:
        muscular, muscle_load = "moderate", "full"
    elif upper:
        muscular, muscle_load = "moderate", "upper"
    else:
        muscular, muscle_load = "low", "none"
    duration_cost = "low" if duration < 30 else "high" if duration >= 75 else "moderate"
    systemic = "moderate" if muscular == "high" or duration >= 60 else "low"
    evidence = [
        f"{duration:.0f} min strength session",
        f"working sets by region: lower {lower}, upper {upper}, full {full}",
    ]
    if context.get("near_recent_high_exercises"):
        evidence.append(
            "loads were near recent working-set highs for "
            + ", ".join(context["near_recent_high_exercises"])
        )
    unknown = int(working.get("unknown", 0))
    if unknown:
        evidence.append(f"{unknown} working sets have unknown muscle classification")
    confidence = "high" if sum(counts.values()) >= 2 and unknown == 0 else "medium" if sum(counts.values()) else "low"
    summary = dict(context)
    return SessionCost(
        int(row["id"]), "strength_local", systemic, "low", muscular, muscle_load,
        "strength", "moderate" if muscular == "high" else "low", duration_cost,
        confidence, tuple(evidence), (), summary,
    )


def _strength_context(connection, row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("sport")) != "strength":
        return {}
    activity_id = int(row["id"])
    sets = [dict(item) for item in connection.execute(
        "SELECT exercise_name, set_number, repetitions, load_value, load_unit FROM strength_sets WHERE activity_id = ?",
        (activity_id,),
    )]
    settings = config.get("coaching", {}).get("cost_model", {}).get("strength", {})
    fraction = float(settings.get("working_set_load_fraction", 0.60))
    regions: dict[str, str] = {}
    max_by_exercise: dict[str, float] = defaultdict(float)
    for item in sets:
        name = _normalize_exercise(item["exercise_name"])
        regions[name] = _exercise_region(name, settings)
        max_by_exercise[name] = max(max_by_exercise[name], _number(item.get("load_value")) or 0)
    movement_counts: Counter[str] = Counter(regions.values())
    working_sets: Counter[str] = Counter()
    for item in sets:
        name = _normalize_exercise(item["exercise_name"])
        load = _number(item.get("load_value")) or 0
        reps = int(item.get("repetitions") or 0)
        maximum = max_by_exercise[name]
        if reps > 0 and (maximum <= 0 or load >= maximum * fraction):
            working_sets[regions[name]] += 1

    start = datetime.fromisoformat(str(row["started_at"]))
    history_start = start - timedelta(days=int(settings.get("history_days", 90)))
    historical: dict[str, float] = defaultdict(float)
    for item in connection.execute(
        """SELECT s.exercise_name, MAX(s.load_value) AS max_load
           FROM strength_sets s JOIN activities a ON a.id = s.activity_id
           WHERE a.started_at >= ? AND a.started_at < ? AND a.id != ?
           GROUP BY s.exercise_name""",
        (history_start.isoformat(), start.isoformat(), activity_id),
    ):
        historical[_normalize_exercise(item["exercise_name"])] = float(item["max_load"] or 0)
    near = [
        name for name, load in max_by_exercise.items()
        if historical.get(name, 0) > 0 and load >= historical[name] * 0.90
    ]
    return {
        "movement_counts": dict(movement_counts),
        "working_sets": dict(working_sets),
        "exercise_max_loads": dict(max_by_exercise),
        "historical_max_loads": dict(historical),
        "near_recent_high": bool(near),
        "near_recent_high_exercises": sorted(near),
    }


def _exercise_region(name: str, settings: dict[str, Any]) -> str:
    aliases = settings.get("exercise_aliases", {})
    for region in ("lower_body", "full_body", "core", "upper_body"):
        if any(_normalize_exercise(pattern) in name for pattern in aliases.get(region, [])):
            return {"lower_body": "lower", "full_body": "full", "upper_body": "upper"}.get(region, "core")
    return "unknown"


def _normalize_exercise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _persist(connection, result: SessionCost) -> None:
    connection.execute(
        """INSERT INTO session_costs (
               activity_id, cost_class, systemic_cost, cardiovascular_cost, muscular_cost,
               muscle_load, sport_specific_cost, intensity_cost, duration_cost, confidence,
               evidence_json, context_flags_json, exercise_summary_json, model_version, classified_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(activity_id) DO UPDATE SET
               cost_class=excluded.cost_class, systemic_cost=excluded.systemic_cost,
               cardiovascular_cost=excluded.cardiovascular_cost, muscular_cost=excluded.muscular_cost,
               muscle_load=excluded.muscle_load, sport_specific_cost=excluded.sport_specific_cost,
               intensity_cost=excluded.intensity_cost, duration_cost=excluded.duration_cost,
               confidence=excluded.confidence, evidence_json=excluded.evidence_json,
               context_flags_json=excluded.context_flags_json,
               exercise_summary_json=excluded.exercise_summary_json,
               model_version=excluded.model_version, classified_at=excluded.classified_at""",
        (
            result.activity_id, result.cost_class, result.systemic_cost,
            result.cardiovascular_cost, result.muscular_cost, result.muscle_load,
            result.sport_specific_cost, result.intensity_cost, result.duration_cost,
            result.confidence, json.dumps(result.evidence, ensure_ascii=False),
            json.dumps(result.context_flags, ensure_ascii=False),
            json.dumps(result.exercise_summary, ensure_ascii=False), MODEL_VERSION,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _upper_zone_fraction(raw: Any) -> float:
    try:
        values = [float(value or 0) for value in json.loads(raw or "[]")]
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    total = sum(values)
    return sum(values[-2:]) / total if total and len(values) >= 2 else 0.0


def _max_level(*values: str) -> str:
    return max(values, key=lambda value: LEVELS.get(value, 0))


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_COST_QUERY = """
SELECT a.*, d.normalized_power_w, d.intensity_factor, d.estimated_tss,
       d.selected_load, d.load_method, d.efficiency_factor,
       d.aerobic_decoupling_pct, d.late_fade_pct, d.pace_seconds_per_km,
       d.hr_coverage, d.power_zones_json, d.hr_zones_json, d.ftp_w,
       p.title AS planned_title, p.session_type AS planned_type,
       p.intensity AS planned_intensity, p.description AS planned_description,
       (SELECT AVG(s.temperature_c) FROM activity_streams s
        WHERE s.activity_id = a.id AND s.temperature_c IS NOT NULL) AS avg_temperature_c,
       (SELECT MAX(s.temperature_c) FROM activity_streams s
        WHERE s.activity_id = a.id AND s.temperature_c IS NOT NULL) AS max_temperature_c
FROM activities a
LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
LEFT JOIN planned_activity_matches m ON m.activity_id = a.id AND m.is_selected = 1
LEFT JOIN planned_sessions p ON p.id = m.planned_session_id
"""
