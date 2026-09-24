from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from endurance_lab.coaching_models import SessionClassification
from endurance_lab.config import load_athlete_config
from endurance_lab.db import connect, init_db
from endurance_lab.session_cost import assess_session_cost, classify_session_costs, session_cost


CLASSIFIER_VERSION = 1
QUALITY_CLASSES = {"TEMPO", "THRESHOLD", "VO2", "INTERVAL", "RACE", "SWIM_QUALITY"}


def classify_activity(
    activity_id: int | str,
    database: str | Path | None = None,
    persist: bool = True,
) -> SessionClassification:
    init_db(database)
    cost = session_cost(activity_id, database, persist=persist)
    with connect(database) as connection:
        row = connection.execute(_ACTIVITY_QUERY + " WHERE a.id = ?", (cost.activity_id,)).fetchone()
        if not row:
            raise ValueError(f"Activity {activity_id} was not found")
        result = classify_record(dict(row), load_athlete_config(), cost=cost)
        if persist:
            _persist(connection, result)
            connection.commit()
    return result


def classify_activities(
    database: str | Path | None = None,
    before: str | None = None,
    persist: bool = True,
) -> list[SessionClassification]:
    init_db(database)
    costs = {item.activity_id: item for item in classify_session_costs(database, before, persist=persist)}
    with connect(database) as connection:
        query = _ACTIVITY_QUERY
        params: tuple[Any, ...] = ()
        if before:
            query += " WHERE a.started_at < ?"
            params = (before,)
        query += " ORDER BY a.started_at"
        rows = connection.execute(query, params).fetchall()
        config = load_athlete_config()
        results = [classify_record(dict(row), config, cost=costs[int(row["id"])]) for row in rows]
        if persist:
            for result in results:
                _persist(connection, result)
            connection.commit()
    return results


def classify_record(
    row: dict[str, Any],
    config: dict[str, Any] | None = None,
    lower_body_sets: int = 0,
    cost=None,
) -> SessionClassification:
    config = config or load_athlete_config()
    sport = str(row.get("sport") or "other").lower()
    duration = float(
        row.get("moving_seconds") or row.get("elapsed_seconds") or row.get("duration_seconds") or 0
    )
    minutes = duration / 60
    selected_load = _number(row.get("selected_load")) or 0.0
    intensity_factor = _number(row.get("intensity_factor"))
    name = str(row.get("name") or "").lower()
    planned = " ".join(
        str(row.get(key) or "") for key in ("planned_title", "planned_type", "planned_intensity", "planned_description")
    ).lower()
    combined = f"{name} {planned}"
    hard_load = float(config.get("load_model", {}).get("hard_session_load", 100))
    long_minutes = config.get("load_model", {}).get("long_session_minutes", {})
    hr_high = _upper_zone_fraction(row.get("hr_zones_json"))
    power_high = _upper_zone_fraction(row.get("power_zones_json"))
    reasons: list[str] = []
    evidence_types = 0
    cost = cost or assess_session_cost(
        row, config,
        {"movement_counts": {"lower": int(lower_body_sets > 0)}, "working_sets": {"lower": lower_body_sets}}
        if sport == "strength" else {},
    )

    if sport == "strength":
        muscular = cost.muscular_cost
        identified_lower_sets = int(cost.exercise_summary.get("working_sets", {}).get("lower", lower_body_sets))
        reasons.append(
            f"strength activity with {identified_lower_sets} identifiable lower-body working sets"
            if identified_lower_sets else "strength activity with no identifiable lower-body working sets"
        )
        stress = cost.systemic_cost
        return _result(row, "STRENGTH", "high", stress, muscular, reasons)

    if sport == "swimming":
        quality_words = _contains(combined, "interval", "threshold", "sprint", "tempo", "quality")
        if quality_words and (hr_high >= 0.10 or selected_load >= hard_load * 0.5):
            classification, confidence = "SWIM_QUALITY", "high"
            reasons.append("quality wording supported by measured intensity")
        elif minutes >= float(long_minutes.get("swimming", 60)):
            classification, confidence = "SWIM_ENDURANCE", "medium"
            reasons.append("duration meets configured long-swim threshold")
        elif selected_load and selected_load < hard_load * 0.35:
            classification, confidence = "SWIM_EASY", "medium"
            reasons.append("low measured session load")
        else:
            classification, confidence = "SWIM_ENDURANCE", "low"
            reasons.append("swim data does not distinguish easy from quality work")
        return _result(row, classification, confidence, cost.systemic_cost, cost.muscular_cost, reasons)

    if sport not in {"cycling", "running"}:
        if minutes <= 10 and selected_load < hard_load * 0.2:
            return _result(row, "REST", "medium", cost.systemic_cost, cost.muscular_cost, ["very short, low-load activity"])
        if _contains(combined, "recovery", "walk", "yoga", "mobility") or selected_load < hard_load * 0.35:
            return _result(row, "RECOVERY", "medium", cost.systemic_cost, cost.muscular_cost, ["low-load recovery-compatible activity"])
        return _result(row, "OTHER", "low", cost.systemic_cost, cost.muscular_cost, ["activity type is not coaching-specific"])

    planned_quality = _quality_from_text(planned)
    title_quality = _quality_from_text(name)
    metric_quality = _quality_from_metrics(intensity_factor, hr_high, power_high)
    if planned_quality:
        evidence_types += 1
        reasons.append(f"matched plan describes {planned_quality.lower()} work")
    if title_quality:
        evidence_types += 1
        reasons.append(f"activity title suggests {title_quality.lower()} work")
    if metric_quality:
        evidence_types += 1
        reasons.append(metric_quality[1])

    classification = planned_quality or (metric_quality[0] if metric_quality else None) or title_quality
    if classification is None:
        long_threshold = float(long_minutes.get(sport, 120 if sport == "cycling" else 75))
        if minutes >= long_threshold:
            classification = "LONG_ENDURANCE"
            reasons.append(f"duration {minutes:.0f} min meets configured long-{sport} threshold")
            evidence_types += 1
        elif intensity_factor is not None and intensity_factor < 0.60:
            classification = "RECOVERY"
            reasons.append(f"intensity factor {intensity_factor:.2f} indicates low intensity")
            evidence_types += 1
        elif intensity_factor is not None and intensity_factor < 0.75:
            classification = "EASY_ENDURANCE"
            reasons.append(f"intensity factor {intensity_factor:.2f} is endurance-compatible")
            evidence_types += 1
        else:
            classification = "ENDURANCE"
            reasons.append("no defensible quality-session signal")

    if "race" in combined and (
        planned_quality == "RACE" or selected_load >= hard_load or (intensity_factor or 0) >= 0.9
    ):
        classification = "RACE"
        reasons.append("race wording is supported by high measured stress or the matched plan")

    high_stress = (
        classification in {"THRESHOLD", "VO2", "INTERVAL", "RACE"}
        or selected_load >= hard_load
        or (intensity_factor or 0) >= 0.90
        or max(hr_high, power_high) >= 0.20
    )
    moderate_stress = (
        classification in {"TEMPO", "LONG_ENDURANCE"}
        or selected_load >= hard_load * 0.5
        or (intensity_factor or 0) >= 0.75
    )
    stress = cost.systemic_cost
    muscular = cost.muscular_cost
    confidence = "high" if evidence_types >= 2 else "medium" if evidence_types == 1 else "low"
    return _result(row, classification, confidence, stress, muscular, reasons)


def _quality_from_text(value: str) -> str | None:
    mapping = (
        ("RACE", ("race", "racing", "event")),
        ("VO2", ("vo2", "anaerobic")),
        ("THRESHOLD", ("threshold", "treshold", "ftp")),
        ("TEMPO", ("tempo", "sweet spot", "sweetspot")),
        ("INTERVAL", ("interval", "repeats", "fartlek")),
    )
    return next((label for label, words in mapping if _contains(value, *words)), None)


def _quality_from_metrics(
    intensity_factor: float | None,
    hr_high: float,
    power_high: float,
) -> tuple[str, str] | None:
    if intensity_factor is not None and intensity_factor >= 1.02:
        return "VO2", f"intensity factor {intensity_factor:.2f} is very high"
    if intensity_factor is not None and intensity_factor >= 0.90:
        return "THRESHOLD", f"intensity factor {intensity_factor:.2f} supports threshold stress"
    if max(hr_high, power_high) >= 0.25:
        return "INTERVAL", f"{max(hr_high, power_high):.0%} of zoned time was in upper zones"
    if intensity_factor is not None and intensity_factor >= 0.78:
        return "TEMPO", f"intensity factor {intensity_factor:.2f} supports tempo stress"
    return None


def _upper_zone_fraction(raw: Any) -> float:
    try:
        values = [float(value or 0) for value in json.loads(raw or "[]")]
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    total = sum(values)
    return sum(values[-2:]) / total if total > 0 and len(values) >= 2 else 0.0


def _lower_body_sets(connection, activity_id: int) -> int:
    tokens = ("squat", "deadlift", "leg", "lunge", "calf", "glute", "hip", "hamstring", "quad")
    rows = connection.execute(
        "SELECT exercise_name FROM strength_sets WHERE activity_id = ?", (activity_id,)
    ).fetchall()
    return sum(any(token in str(row[0]).lower() for token in tokens) for row in rows)


def _persist(connection, result: SessionClassification) -> None:
    connection.execute(
        """INSERT INTO activity_classifications (
               activity_id, classification, confidence, stress_level, muscular_load,
               reasons_json, model_version, classified_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(activity_id) DO UPDATE SET
               classification = excluded.classification, confidence = excluded.confidence,
               stress_level = excluded.stress_level, muscular_load = excluded.muscular_load,
               reasons_json = excluded.reasons_json, model_version = excluded.model_version,
               classified_at = excluded.classified_at""",
        (
            result.activity_id, result.classification, result.confidence, result.stress_level,
            result.muscular_load, json.dumps(result.reasons, ensure_ascii=False),
            CLASSIFIER_VERSION, datetime.now(timezone.utc).isoformat(),
        ),
    )


def _result(row, classification, confidence, stress, muscular, reasons) -> SessionClassification:
    return SessionClassification(
        int(row["id"]), classification, confidence, stress, muscular, tuple(reasons)
    )


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _contains(value: str, *terms: str) -> bool:
    return any(term in value for term in terms)


_ACTIVITY_QUERY = """
SELECT a.*, d.normalized_power_w, d.intensity_factor, d.estimated_tss,
       d.selected_load, d.load_method, d.efficiency_factor,
       d.aerobic_decoupling_pct, d.late_fade_pct, d.pace_seconds_per_km,
       d.hr_coverage, d.power_zones_json, d.hr_zones_json,
       p.title AS planned_title, p.session_type AS planned_type,
       p.intensity AS planned_intensity, p.description AS planned_description
FROM activities a
LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
LEFT JOIN planned_activity_matches m ON m.activity_id = a.id AND m.is_selected = 1
LEFT JOIN planned_sessions p ON p.id = m.planned_session_id
"""
