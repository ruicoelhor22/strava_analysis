from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from endurance_lab.coaching_models import AthleteState
from endurance_lab.config import load_athlete_config


@dataclass(frozen=True)
class RuleDecision:
    action: str
    confidence: str
    reasons: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    rules: tuple[str, ...]


def evaluate_rules(session: dict[str, Any], state: AthleteState) -> RuleDecision:
    settings = load_athlete_config().get("coaching", {}).get("rules", {})
    hard = planned_stress(session) == "high"
    key = str(session.get("priority") or "").startswith("A")
    sport = str(session.get("sport") or "other")
    subjective_high = int(settings.get("subjective_high_threshold", 4))
    checkin = state.subjective or {}
    fatigue = checkin.get("fatigue")
    soreness = checkin.get("leg_soreness")
    if fatigue is not None and soreness is not None and fatigue >= 5 and soreness >= 5:
        action = "MOVE_SESSION" if key else "REST"
        return _decision(
            action, "medium", "subjective_high_fatigue_and_soreness",
            "High optional fatigue and leg-soreness reports make automatic hard training inappropriate.",
            {"fatigue": fatigue, "leg_soreness": soreness, "checkin_date": checkin.get("checkin_date")},
        )
    if not hard and fatigue is not None and fatigue >= 5:
        return _decision(
            "RECOVERY_SESSION", "medium", "subjective_high_fatigue",
            "Optional fatigue feedback supports replacing structured training with very easy recovery movement.",
            {"fatigue": fatigue, "checkin_date": checkin.get("checkin_date")},
        )
    if hard and any(value is not None and value >= subjective_high for value in (fatigue, soreness)):
        return _decision(
            "REDUCE_INTENSITY", "medium", "subjective_recovery_watch",
            "Optional subjective feedback indicates a conservative intensity reduction.",
            {"fatigue": fatigue, "leg_soreness": soreness, "threshold": subjective_high},
        )

    consecutive = int(state.recovery_context.get("consecutive_training_days") or 0)
    consecutive_limit = int(settings.get("consecutive_training_days_watch", 6))
    if hard and consecutive >= consecutive_limit + 1:
        return _decision(
            "MOVE_SESSION", "high", "excessive_consecutive_training_days",
            f"The session follows {consecutive} consecutive training days.",
            {"consecutive_training_days": consecutive, "configured_watch": consecutive_limit},
        )

    days_since_hard = state.recovery_context.get(f"days_since_hard_{_sport_label(sport)}")
    hours_since_hard = state.recovery_context.get(f"hours_since_hard_{_sport_label(sport)}")
    spacing_hours = float(settings.get("high_stress_recovery_hours", {}).get(sport, 36))
    hard_spacing_by_day_end = hours_since_hard + 24 if hours_since_hard is not None else None
    if hard and hard_spacing_by_day_end is not None and hard_spacing_by_day_end <= spacing_hours:
        return _decision(
            "MOVE_SESSION", "high", "high_stress_spacing",
            f"Another high-stress {sport} session is within the configured spacing even if today's workout is late.",
            {
                "days_since_hard_session": days_since_hard,
                "hours_since_hard_session": hours_since_hard,
                "maximum_spacing_by_end_of_day": hard_spacing_by_day_end,
                "minimum_spacing_hours": spacing_hours,
            },
        )

    leg_days = state.recovery_context.get("days_since_leg_strength")
    leg_hours = state.recovery_context.get("hours_since_leg_strength")
    leg_spacing = float(settings.get("leg_strength_interference_hours", 36))
    leg_spacing_by_day_end = leg_hours + 24 if leg_hours is not None else None
    if (
        hard and sport in {"cycling", "running"}
        and leg_spacing_by_day_end is not None and leg_spacing_by_day_end <= leg_spacing
    ):
        return _decision(
            "REDUCE_INTENSITY", "high", "leg_strength_interference",
            "Demanding leg strength remains inside the configured spacing even if today's endurance workout is late.",
            {
                "days_since_leg_strength": leg_days,
                "hours_since_leg_strength": leg_hours,
                "maximum_spacing_by_end_of_day": leg_spacing_by_day_end,
                "configured_spacing_hours": leg_spacing,
            },
        )

    load_dimension = state.dimensions["LOAD_CONTEXT"].state
    if hard and load_dimension == "concern":
        return _decision(
            "REDUCE_DURATION", "high", "unusually_high_recent_load",
            "Recent estimated load is unusually high relative to the 28-day weekly reference.",
            {"load_ratio": state.load.get("recent_to_reference_ratio")},
        )

    volume_ratio = state.adherence.get("actual_to_planned_duration_ratio_7d")
    volume_watch = float(settings.get("recent_volume_ratio_watch", 1.30))
    duration = float(session.get("planned_duration_seconds") or 0)
    planned_coverage = int(state.adherence.get("planned_sessions_7d") or 0)
    if (
        not key
        and sport in {"cycling", "running", "swimming"}
        and duration >= 3600
        and planned_coverage >= 3
        and volume_ratio is not None
        and volume_ratio >= volume_watch
    ):
        return _decision(
            "REDUCE_DURATION", "medium", "actual_volume_above_plan",
            "Actual seven-day duration is materially above planned duration.",
            {
                "actual_to_planned_duration_ratio_7d": volume_ratio,
                "threshold": volume_watch,
                "planned_sessions_7d": planned_coverage,
            },
        )

    return RuleDecision(
        "KEEP", "high" if state.data_quality.get("confidence") != "low" else "medium",
        ("No rule has sufficient evidence to change the planned session.",),
        ({
            "load_context": state.dimensions["LOAD_CONTEXT"].state,
            "recovery_spacing": state.dimensions["RECOVERY_SPACING"].state,
            "plan_adherence": state.dimensions["PLAN_ADHERENCE"].state,
        },),
        ("plan_is_anchor",),
    )


def planned_stress(session: dict[str, Any]) -> str:
    sport = str(session.get("sport") or "other").lower()
    text = " ".join(
        str(session.get(key) or "")
        for key in ("title", "session_type", "intensity", "description", "interval_structure")
    ).lower()
    if sport == "strength":
        if "heavy" in text or re.search(r"\brpe\s*(?:8|9|10)\b", text):
            return "high"
        return "moderate" if float(session.get("planned_duration_seconds") or 0) >= 1800 else "low"
    if sport == "other":
        return "low"
    if any(token in text for token in ("threshold", "treshold", "vo2", "interval", "race", "max")):
        return "high"
    if any(token in text for token in ("tempo", "sweet spot", "long")):
        return "moderate"
    return "low"


def _decision(action, confidence, rule, reason, evidence) -> RuleDecision:
    return RuleDecision(action, confidence, (reason,), (evidence,), (rule,))


def _sport_label(sport: str) -> str:
    return {"cycling": "ride", "running": "run", "swimming": "swim"}.get(sport, sport)
