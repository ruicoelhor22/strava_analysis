from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from endurance_lab.coaching_models import ExecutionEvaluation
from endurance_lab.db import connect, init_db
from endurance_lab.interval_analysis import analyze_intervals
from endurance_lab.session_classifier import QUALITY_CLASSES, classify_activity


def evaluate_activity(
    activity_id: int,
    database: str | Path | None = None,
    persist: bool = True,
) -> ExecutionEvaluation:
    init_db(database)
    classification = classify_activity(activity_id, database, persist=persist)
    with connect(database) as connection:
        row = connection.execute(
            """SELECT a.id, a.sport AS actual_sport,
                      COALESCE(a.moving_seconds, a.elapsed_seconds) actual_duration,
                      a.distance_m actual_distance, a.avg_hr, a.avg_power_w,
                      d.normalized_power_w, d.aerobic_decoupling_pct, d.late_fade_pct,
                      p.id planned_session_id, p.sport AS planned_sport, p.planned_duration_seconds,
                      p.planned_distance_m, p.intensity, p.title
               FROM activities a
               LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
               LEFT JOIN planned_activity_matches m ON m.activity_id = a.id AND m.is_selected = 1
               LEFT JOIN planned_sessions p ON p.id = m.planned_session_id
               WHERE a.id = ?""",
            (activity_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Activity {activity_id} was not found")
        dimensions: dict[str, str] = {
            "duration": _ratio_class(row["actual_duration"], row["planned_duration_seconds"]),
            "distance": _ratio_class(row["actual_distance"], row["planned_distance_m"]),
            "intensity": "unknown",
            "interval_completion": "unknown",
            "fade": _fade_class(row["late_fade_pct"]),
            "decoupling": _decoupling_class(row["aerobic_decoupling_pct"]),
            "session_compatibility": (
                "compatible" if row["planned_sport"] == row["actual_sport"] else "different_sport"
                if row["planned_session_id"] else "unknown"
            ),
        }
        evidence: list[str] = []
        intervals: tuple[dict[str, Any], ...] = ()
        interval_result = None
        if (
            classification.classification in QUALITY_CLASSES
            and dimensions["session_compatibility"] == "compatible"
        ):
            interval_result = analyze_intervals(activity_id, database, persist=persist)
            intervals = tuple(interval_result["intervals"])
            expected = interval_result["expected_intervals"]
            detected = interval_result["detected_intervals"]
            if expected:
                dimensions["interval_completion"] = (
                    "complete" if detected >= expected else "partial" if detected else "unknown"
                )
                evidence.append(f"detected {detected} of {expected} expected work intervals")
            target_states = [item["target_adherence"] for item in intervals if item["target_adherence"] != "unknown"]
            if target_states:
                close = target_states.count("close") / len(target_states)
                dimensions["intensity"] = "close" if close >= 0.67 else (
                    "harder" if target_states.count("above") > target_states.count("below") else "easier"
                )
                evidence.append(f"{target_states.count('close')} of {len(target_states)} measured intervals were near target")

        if not row["planned_session_id"]:
            status, confidence = "unknown", "low"
            evidence.append("activity has no selected planned-session match")
        elif dimensions["session_compatibility"] == "different_sport":
            status, confidence = "unknown", "high"
            evidence.append(
                f"actual sport {row['actual_sport']} differs from planned sport {row['planned_sport']}; "
                "the calendar link is retained but execution targets are not compared"
            )
        else:
            status, confidence = _execution_status(dimensions, classification, interval_result)
            evidence.extend(_dimension_evidence(dimensions))
        result = ExecutionEvaluation(
            activity_id, row["planned_session_id"], status, confidence,
            dimensions, tuple(evidence), intervals,
        )
        if persist:
            connection.execute(
                """INSERT INTO workout_evaluations (
                       activity_id, planned_session_id, execution_status, confidence,
                       dimensions_json, evidence_json, evaluated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(activity_id) DO UPDATE SET
                       planned_session_id = excluded.planned_session_id,
                       execution_status = excluded.execution_status,
                       confidence = excluded.confidence,
                       dimensions_json = excluded.dimensions_json,
                       evidence_json = excluded.evidence_json,
                       evaluated_at = excluded.evaluated_at""",
                (
                    activity_id, row["planned_session_id"], status, confidence,
                    json.dumps(dimensions), json.dumps(evidence, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
    return result


def _execution_status(dimensions, classification, interval_result) -> tuple[str, str]:
    duration = dimensions["duration"]
    intensity = dimensions["intensity"]
    completion = dimensions["interval_completion"]
    if duration == "far_under" and completion in {"partial", "unknown"}:
        return "failed", "medium" if interval_result and interval_result["confidence"] != "low" else "low"
    if duration in {"under", "far_under"} or completion == "partial":
        return "partial", "medium"
    if intensity == "harder" or duration == "over":
        return "harder_than_planned", "medium"
    if intensity == "easier":
        return "easier_than_planned", "medium"
    if duration == "close" and (completion in {"complete", "unknown"}):
        return "successful", "high" if completion == "complete" else "medium"
    return "unknown", "low"


def _ratio_class(actual, planned) -> str:
    if actual is None or not planned:
        return "unknown"
    ratio = float(actual) / float(planned)
    return "far_under" if ratio < 0.5 else "under" if ratio < 0.85 else "over" if ratio > 1.15 else "close"


def _fade_class(value) -> str:
    if value is None:
        return "unknown"
    return "substantial" if float(value) > 10 else "moderate" if float(value) > 5 else "limited"


def _decoupling_class(value) -> str:
    if value is None:
        return "unknown"
    return "high" if abs(float(value)) > 8 else "moderate" if abs(float(value)) > 5 else "limited"


def _dimension_evidence(dimensions) -> list[str]:
    return [f"{name.replace('_', ' ')}: {value}" for name, value in dimensions.items() if value != "unknown"]
