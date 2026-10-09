from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from endurance_lab.metadata_activities import MetadataActivity, upsert_metadata_activity
from endurance_lab.session_conclusion import build_session_conclusion, load_session_conclusion


def test_hard_ride_conclusion_uses_cost_and_reliable_plan_execution():
    conclusion = build_session_conclusion(
        {"sport": "cycling", "metadata_only": 0},
        {
            "cardiovascular_cost": "high", "muscular_cost": "moderate",
            "intensity_cost": "high", "duration_cost": "moderate",
            "muscle_load": "lower", "confidence": "high",
        },
        {"planned_session_id": 1, "execution_status": "successful", "confidence": "medium"},
    )
    assert conclusion.headline == "A demanding high-intensity session."
    assert "close to the linked plan" in conclusion.explanation
    assert conclusion.caveat is None


def test_strength_conclusion_prioritizes_leg_cost_over_cardio():
    conclusion = build_session_conclusion(
        {"sport": "strength", "metadata_only": 0},
        {
            "cardiovascular_cost": "low", "muscular_cost": "high",
            "intensity_cost": "moderate", "duration_cost": "moderate",
            "muscle_load": "lower", "confidence": "high",
        },
    )
    assert "lower-body strength" in conclusion.headline
    assert "not cardiovascular stress" in conclusion.explanation


def test_weak_or_missing_evidence_does_not_claim_success():
    activity = {"sport": "cycling", "metadata_only": 0}
    assert "not been assessed" in build_session_conclusion(activity).headline
    conclusion = build_session_conclusion(
        activity,
        {
            "cardiovascular_cost": "low", "muscular_cost": "low",
            "intensity_cost": "low", "duration_cost": "low",
            "muscle_load": "lower", "confidence": "low",
        },
        {"planned_session_id": 1, "execution_status": "successful", "confidence": "low"},
    )
    assert "linked plan" not in conclusion.explanation
    assert "Low confidence" in conclusion.caveat


def test_metadata_only_conclusion_is_explicitly_limited():
    database = Path("data/test-runs") / uuid4().hex / "conclusion.sqlite3"
    activity_id = upsert_metadata_activity(
        MetadataActivity("test-summary", datetime(2026, 9, 24, tzinfo=timezone.utc), "cycling"),
        database,
    )
    conclusion = load_session_conclusion(activity_id, database)
    assert conclusion is not None
    assert "Summary-only" in conclusion.headline
    assert "sensor stream was not available" in conclusion.explanation
