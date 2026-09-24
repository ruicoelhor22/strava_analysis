from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlanTarget:
    target_type: str
    raw_text: str
    minimum_value: float | None = None
    maximum_value: float | None = None
    unit: str | None = None
    confidence: str = "low"


@dataclass(frozen=True)
class PlannedSession:
    source_key: str
    source_row: int
    planned_date: date
    week_start: date
    phase: str
    week_type: str | None
    sport: str
    session_type: str | None
    workout_code: str | None
    title: str
    description: str | None
    planned_duration_seconds: float | None
    planned_distance_m: float | None
    intensity: str | None
    interval_structure: str | None
    priority: str | None
    notes: str | None
    source_status: str | None
    workbook_actual: dict[str, Any]
    targets: tuple[PlanTarget, ...]
    raw_source: dict[str, Any]


@dataclass(frozen=True)
class TrainingWeek:
    week_start: date
    week_end: date
    phase: str
    week_type: str | None
    objectives: str | None
    raw_source: dict[str, Any]


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    start_date: date
    end_date: date
    objectives: str | None
    raw_source: dict[str, Any]


@dataclass(frozen=True)
class ParsedTrainingPlan:
    source: Path
    name: str
    source_sha256: str
    date_start: date
    date_end: date
    sheets: tuple[str, ...]
    phases: tuple[TrainingPhase, ...]
    weeks: tuple[TrainingWeek, ...]
    sessions: tuple[PlannedSession, ...]
    workbook_snapshot: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class PlanImportSummary:
    workbook: Path
    unchanged: bool
    sheets_inspected: int
    phases: int
    weeks: int
    sessions: int
    date_start: date
    date_end: date
    sports: dict[str, int]
    warnings: tuple[str, ...] = ()

