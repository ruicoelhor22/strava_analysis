from __future__ import annotations

import pandas as pd

from dashboard import app
from endurance_lab.strength_profile import summarize_strength


def test_gym_summary_works_with_previous_loaded_summarizer(monkeypatch):
    # A Streamlit rerun can retain an older imported helper without the new kwarg.
    def previous_signature(rows):
        summary = summarize_strength(rows)
        summary.pop("counted_sets")
        for exercise in summary["exercises"]:
            exercise.pop("counted_sets")
            for item in exercise["sets"]:
                item.pop("load_recorded")
        return summary

    monkeypatch.setattr(app, "summarize_strength", previous_signature)
    frame = pd.DataFrame([
        {"exercise_name": "LAT_PULLDOWN", "set_number": 1, "repetitions": 10, "load_value": 60, "load_unit": None},
        {"exercise_name": "LAT_PULLDOWN", "set_number": 2, "repetitions": 8, "load_value": 80, "load_unit": None},
        {"exercise_name": "LAT_PULLDOWN", "set_number": 3, "repetitions": 8, "load_value": 80, "load_unit": None},
    ])
    result = app._gym_summary(frame)
    assert result["volume_kg"] == 1880
    assert result["counted_sets"] == 3
    assert [item["load_kg"] for item in result["exercises"][0]["sets"]] == [60, 80, 80]
    assert frame["load_unit"].isna().all()  # presentation must not rewrite stored data
