from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Trackpoint:
    sequence: int
    recorded_at: datetime | None = None
    elapsed_seconds: float | None = None
    distance_m: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    heart_rate: float | None = None
    cadence: float | None = None
    speed_mps: float | None = None
    power_w: float | None = None
    temperature_c: float | None = None
    moving: bool | None = None


@dataclass
class Lap:
    lap_number: int
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    distance_m: float | None = None
    ascent_m: float | None = None
    descent_m: float | None = None
    calories: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    avg_speed_mps: float | None = None
    max_speed_mps: float | None = None
    avg_cadence: float | None = None
    max_cadence: float | None = None
    avg_power_w: float | None = None
    max_power_w: float | None = None


@dataclass
class StrengthSet:
    exercise_name: str
    set_number: int
    repetitions: int | None = None
    load_value: float | None = None
    load_unit: str | None = None
    started_at: datetime | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedActivity:
    stable_id: str
    source_activity_id: str | None
    sport: str
    started_at: datetime
    ended_at: datetime | None = None
    name: str | None = None
    elapsed_seconds: float | None = None
    moving_seconds: float | None = None
    distance_m: float | None = None
    ascent_m: float | None = None
    descent_m: float | None = None
    calories: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    avg_speed_mps: float | None = None
    max_speed_mps: float | None = None
    avg_cadence: float | None = None
    max_cadence: float | None = None
    avg_power_w: float | None = None
    max_power_w: float | None = None
    device: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    laps: list[Lap] = field(default_factory=list)
    trackpoints: list[Trackpoint] = field(default_factory=list)
    strength_sets: list[StrengthSet] = field(default_factory=list)


# Backward-compatible name used by the Phase 1 TCX tests and importer.
Activity = ParsedActivity
