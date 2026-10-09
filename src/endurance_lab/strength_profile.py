"""Display-oriented summaries of recorded strength sets.

Volume is external load moved, not a measure of training stress. Body weight,
assistance, and unilateral sides cannot be inferred from these exports.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


def exercise_target(name: str) -> str:
    """Return a conservative primary muscle group from the exercise label."""
    key = name.upper().replace(" ", "_").replace("-", "_")
    if any(part in key for part in ("CRUNCH", "LEG_RAISE", "SIDE_BEND", "PLANK")):
        return "Core"
    if "CLEAN_AND_JERK" in key:
        return "Full body"
    if any(part in key for part in ("ROMANIAN_DEADLIFT", "LEG_CURL", "HAMSTRING")):
        return "Hamstrings"
    if any(part in key for part in ("LEG_PRESS", "LEG_EXTENSION", "SQUAT", "LUNGE")):
        return "Quads"
    if any(part in key for part in ("CHEST_PRESS", "BENCH_PRESS", "FLYE", "PUSH_UP")):
        return "Chest"
    if any(part in key for part in ("LAT_PULL", "LAT_PUL", "PULL_UP", "STRAIGHT_ARM_PULLDOWN")):
        return "Lats"
    if "ROW" in key:
        return "Upper back"
    if any(part in key for part in ("TRICEPS", "TRICEP", "PRESSDOWN")):
        return "Triceps"
    if any(part in key for part in ("BICEPS", "BICEP", "CURL", "PREACHER")):
        return "Biceps"
    if any(part in key for part in ("SHOULDER_PRESS", "OVERHEAD_PRESS", "OVERHEAD_DUMBBELL_PRESS", "LATERAL_RAISE")):
        return "Shoulders"
    return "Unclassified"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _kg_multiplier(unit: Any, device: Any, assume_missing_kg: bool) -> float | None:
    label = str(unit).strip().lower() if unit is not None else ""
    if label in {"nan", "<na>", "none"}:
        label = ""
    if label in {"kg", "kilogram", "kilograms"}:
        return 1.0
    if label in {"lb", "lbs", "pound", "pounds"}:
        return 0.45359237
    # Existing exports predate the athlete's confirmation that their loads are kg.
    if not label and (assume_missing_kg or str(device or "").strip().lower() == "hevy"):
        return 1.0
    return None


def summarize_strength(rows: Iterable[dict[str, Any]], *, assume_missing_kg: bool = False) -> dict[str, Any]:
    exercises: dict[str, dict[str, Any]] = {}
    groups: Counter[str] = Counter()
    total_reps = 0
    excluded = 0
    count = 0
    for row in rows:
        count += 1
        name = str(row.get("exercise_name") or "Unknown exercise")
        exercise = exercises.setdefault(name, {
            "name": name.replace("_", " ").title(),
            "group": exercise_target(name), "sets": [], "volume_kg": 0.0,
            "reps": 0, "counted_sets": 0,
        })
        reps = _number(row.get("repetitions"))
        load = _number(row.get("load_value"))
        multiplier = _kg_multiplier(row.get("load_unit"), row.get("activity_device"), assume_missing_kg)
        volume = reps * load * multiplier if reps is not None and load is not None and multiplier is not None and reps > 0 and load >= 0 else None
        if volume is None:
            excluded += 1
        else:
            exercise["volume_kg"] += volume
            exercise["counted_sets"] += 1
        if reps is not None and reps > 0:
            exercise["reps"] += int(reps)
            total_reps += int(reps)
        groups[exercise["group"]] += 1
        exercise["sets"].append({
            "number": row.get("set_number") or len(exercise["sets"]) + 1,
            "reps": int(reps) if reps is not None else None,
            "load_kg": load * multiplier if load is not None and multiplier is not None else None,
            "load_recorded": load is not None,
            "volume_kg": volume,
        })
    return {
        "sets": count, "reps": total_reps, "exercises": list(exercises.values()),
        "counted_sets": count - excluded,
        "volume_kg": sum(item["volume_kg"] for item in exercises.values()),
        "excluded_sets": excluded, "groups": groups.most_common(),
    }
