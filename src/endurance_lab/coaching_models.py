from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class SessionClassification:
    activity_id: int
    classification: str
    confidence: str
    stress_level: str
    muscular_load: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class SessionCost:
    activity_id: int
    cost_class: str
    systemic_cost: str
    cardiovascular_cost: str
    muscular_cost: str
    muscle_load: str
    sport_specific_cost: str
    intensity_cost: str
    duration_cost: str
    confidence: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    context_flags: tuple[str, ...] = field(default_factory=tuple)
    exercise_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = list(self.evidence)
        result["context_flags"] = list(self.context_flags)
        return result


@dataclass(frozen=True)
class OptionalGate:
    state: str
    acceptable_states: tuple[str, ...]
    fallback: dict[str, Any]
    signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["acceptable_states"] = list(self.acceptable_states)
        result["signals"] = list(self.signals)
        return result


@dataclass(frozen=True)
class TrendSignal:
    sport: str
    state: str
    confidence: str
    evidence: tuple[str, ...]
    comparison_window: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = list(self.evidence)
        return result


@dataclass(frozen=True)
class ContextDimension:
    state: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class AthleteState:
    date: date
    recent_training: dict[str, Any]
    load: dict[str, Any]
    recovery_context: dict[str, Any]
    performance: dict[str, Any]
    adherence: dict[str, Any]
    data_quality: dict[str, Any]
    subjective: dict[str, Any] | None
    dimensions: dict[str, ContextDimension]

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "recent_training": self.recent_training,
            "load": self.load,
            "recovery_context": self.recovery_context,
            "performance": self.performance,
            "adherence": self.adherence,
            "data_quality": self.data_quality,
            "subjective": self.subjective,
            "dimensions": {key: value.to_dict() for key, value in self.dimensions.items()},
        }


@dataclass(frozen=True)
class Prescription:
    date: date
    planned_session_id: int | None
    original: dict[str, Any] | None
    prescribed: dict[str, Any] | None
    action: str
    reasons: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    confidence: str
    rules_triggered: tuple[str, ...]
    prescription_id: int | None = None
    optional_gate: OptionalGate | None = None
    decision_trace: dict[str, Any] = field(default_factory=dict)
    recovery_runway_h: float | None = None
    protected_sessions: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "planned_session_id": self.planned_session_id,
            "original": self.original,
            "prescribed": self.prescribed,
            "action": self.action,
            "reasons": list(self.reasons),
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "rules_triggered": list(self.rules_triggered),
            "prescription_id": self.prescription_id,
            "optional_gate": self.optional_gate.to_dict() if self.optional_gate else None,
            "decision_trace": self.decision_trace,
            "recovery_runway_h": self.recovery_runway_h,
            "protected_sessions": list(self.protected_sessions),
        }


@dataclass(frozen=True)
class ExecutionEvaluation:
    activity_id: int
    planned_session_id: int | None
    execution_status: str
    confidence: str
    dimensions: dict[str, str]
    evidence: tuple[str, ...] = field(default_factory=tuple)
    intervals: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = list(self.evidence)
        result["intervals"] = list(self.intervals)
        return result
