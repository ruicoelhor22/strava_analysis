from __future__ import annotations

import pytest

from endurance_lab.strength_profile import exercise_target, summarize_strength


def test_hevy_legacy_kg_volume_and_muscle_focus():
    summary = summarize_strength([
        {"exercise_name": "LAT_PULLDOWN", "set_number": 1, "repetitions": 10, "load_value": 70, "load_unit": None, "activity_device": "Hevy"},
        {"exercise_name": "LAT_PULLDOWN", "set_number": 2, "repetitions": 8, "load_value": 75, "load_unit": "kg", "activity_device": "Hevy"},
        {"exercise_name": "HANGING_LEG_RAISE", "set_number": 1, "repetitions": 12, "load_value": None, "load_unit": None, "activity_device": "Hevy"},
    ])
    assert summary["volume_kg"] == 1300
    assert summary["sets"] == 3
    assert summary["reps"] == 30
    assert summary["excluded_sets"] == 1
    assert summary["groups"] == [("Lats", 2), ("Core", 1)]
    assert summary["exercises"][0]["volume_kg"] == 1300


def test_explicit_pounds_convert_and_unknown_unit_does_not():
    summary = summarize_strength([
        {"exercise_name": "ROW_GENERIC", "repetitions": 10, "load_value": 100, "load_unit": "lb", "activity_device": "Other"},
        {"exercise_name": "ROW_GENERIC", "repetitions": 10, "load_value": 100, "load_unit": None, "activity_device": "Other"},
    ])
    assert summary["volume_kg"] == pytest.approx(453.59237)
    assert summary["excluded_sets"] == 1


def test_confirmed_kg_covers_legacy_rows_without_device_or_with_nan_unit():
    summary = summarize_strength([
        {"exercise_name": "LAT_PULLDOWN", "repetitions": 10, "load_value": 60, "load_unit": None},
        {"exercise_name": "LAT_PULLDOWN", "repetitions": 8, "load_value": 80, "load_unit": float("nan")},
    ], assume_missing_kg=True)
    assert summary["volume_kg"] == 1240
    assert [item["load_kg"] for item in summary["exercises"][0]["sets"]] == [60, 80]


@pytest.mark.parametrize(("name", "group"), [
    ("LAT_PULDOWN", "Lats"), ("ROW_GENERIC", "Upper back"),
    ("BARBELL_BENCH_PRESS", "Chest"), ("MACHINE_LEG_EXTENSION", "Quads"),
    ("DUMBBELL_ROMANIAN_DEADLIFTS", "Hamstrings"),
    ("HANGING_LEG_RAISE", "Core"), ("OVERHEAD_DUMBBELL_PRESS", "Shoulders"),
    ("CLEAN_AND_JERK", "Full body"),
    ("UNKNOWN_MOVEMENT", "Unclassified"),
])
def test_primary_muscle_group(name, group):
    assert exercise_target(name) == group
