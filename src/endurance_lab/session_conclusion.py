from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from endurance_lab.db import connect, init_db


@dataclass(frozen=True)
class SessionConclusion:
    headline: str
    explanation: str
    caveat: str | None = None


def build_session_conclusion(
    activity: Mapping[str, Any],
    cost: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
) -> SessionConclusion:
    """Summarize persisted, explainable evidence without inventing sensor data."""
    if activity.get("metadata_only"):
        return SessionConclusion(
            "Summary-only session; detailed conclusion unavailable.",
            "The activity is recorded, but its original sensor stream was not available for a reliable effort assessment.",
            "Duration and distance may still contribute to the training history and plan match.",
        )
    if not cost:
        return SessionConclusion(
            "This session has not been assessed yet.",
            "Run the normal sync or recalculate analytics to populate its training-cost evidence.",
        )

    sport = str(activity.get("sport") or "other").lower()
    cardio = str(cost.get("cardiovascular_cost") or "unknown").lower()
    muscular = str(cost.get("muscular_cost") or "unknown").lower()
    intensity = str(cost.get("intensity_cost") or "unknown").lower()
    muscle_load = str(cost.get("muscle_load") or "none").lower()
    duration = str(cost.get("duration_cost") or "unknown").lower()

    if sport == "strength":
        if muscle_load == "lower" and muscular == "high":
            headline = "Meaningful lower-body strength load."
            explanation = "The main recovery consideration is local leg work, not cardiovascular stress."
        elif muscle_load == "lower":
            headline = "Lower-body strength work added local muscle load."
            explanation = f"Muscular cost was {muscular}; cardiovascular cost was {cardio}."
        elif muscle_load == "upper":
            headline = "Upper-body strength work with limited leg interference."
            explanation = f"Muscular cost was {muscular}; cardiovascular cost was {cardio}."
        else:
            headline = "Strength work is recorded, but its local cost is less clear."
            explanation = f"The model estimates {muscular} muscular and {cardio} cardiovascular cost."
    elif cardio == "low" and muscular == "low":
        headline = "A low-cost session."
        explanation = "The available data indicates little cardiovascular or muscular stress."
    elif cardio == "low" and muscular in {"moderate", "high"}:
        headline = "Muscle load matters more than cardiovascular load here."
        explanation = f"Cardiovascular cost was low, while muscular cost was {muscular}."
    elif cardio == "high" and intensity == "high":
        headline = "A demanding high-intensity session."
        explanation = f"Cardiovascular cost was high and muscular cost was {muscular}."
    elif cardio == "high":
        headline = "A substantial cardiovascular session."
        explanation = f"The model estimates high cardiovascular and {muscular} muscular cost."
    else:
        headline = "A moderate training stimulus."
        explanation = f"Cardiovascular cost was {cardio} and muscular cost was {muscular}."

    if duration == "high" and cardio != "low":
        explanation += " Duration adds to the overall load."

    plan_comment = _plan_comment(evaluation)
    if plan_comment:
        explanation += f" {plan_comment}"

    caveat = None
    if str(cost.get("confidence") or "").lower() == "low":
        caveat = "Low confidence: limited measured detail makes this a cautious interpretation."
    return SessionConclusion(headline, explanation, caveat)


def load_session_conclusion(
    activity_id: int, database: str | Path | None = None
) -> SessionConclusion | None:
    init_db(database)
    with connect(database) as connection:
        row = connection.execute(
            """SELECT a.sport, a.metadata_only,
                      c.cardiovascular_cost, c.muscular_cost, c.intensity_cost,
                      c.muscle_load, c.duration_cost, c.confidence AS cost_confidence,
                      e.planned_session_id, e.execution_status, e.confidence AS execution_confidence
               FROM activities a
               LEFT JOIN session_costs c ON c.activity_id = a.id
               LEFT JOIN workout_evaluations e ON e.activity_id = a.id
               WHERE a.id = ?""",
            (int(activity_id),),
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    cost = None if record["cardiovascular_cost"] is None else {
        "cardiovascular_cost": record["cardiovascular_cost"],
        "muscular_cost": record["muscular_cost"],
        "intensity_cost": record["intensity_cost"],
        "muscle_load": record["muscle_load"],
        "duration_cost": record["duration_cost"],
        "confidence": record["cost_confidence"],
    }
    evaluation = None if record["planned_session_id"] is None else {
        "planned_session_id": record["planned_session_id"],
        "execution_status": record["execution_status"],
        "confidence": record["execution_confidence"],
    }
    return build_session_conclusion(record, cost, evaluation)


def _plan_comment(evaluation: Mapping[str, Any] | None) -> str | None:
    if not evaluation or not evaluation.get("planned_session_id"):
        return None
    if str(evaluation.get("confidence") or "").lower() == "low":
        return None
    return {
        "successful": "Measured execution was close to the linked plan.",
        "partial": "Only part of the linked plan was completed.",
        "harder_than_planned": "The session was harder than the linked plan.",
        "easier_than_planned": "The session was easier than the linked plan.",
        "failed": "The linked planned work was not completed.",
    }.get(str(evaluation.get("execution_status") or ""))
