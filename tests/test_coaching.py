from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from uuid import uuid4

from endurance_lab.athlete_state import athlete_state
from endurance_lab.coaching_backtest import backtest
from endurance_lab.db import connect, init_db
from endurance_lab.prescription import prescribe
from endurance_lab.performance_trends import performance_trends
from endurance_lab.session_classifier import classify_activity
from endurance_lab.session_cost import assess_session_cost, session_cost
from endurance_lab.subjective import save_checkin
from endurance_lab.weekly_adjustment import adjust_week
from endurance_lab.workout_execution import evaluate_activity


def test_schema_three_migrates_additively_to_adaptive_coaching_schema():
    root = Path("data/test-runs") / uuid4().hex
    root.mkdir(parents=True)
    database = root / "migration.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_meta VALUES (3)")
        connection.execute(
            """CREATE TABLE workout_prescriptions (
                   id INTEGER PRIMARY KEY, planned_session_id INTEGER,
                   prescribed_date TEXT NOT NULL, sport TEXT NOT NULL, title TEXT NOT NULL,
                   description TEXT, duration_seconds REAL, distance_m REAL, intensity TEXT,
                   targets_json TEXT, reason TEXT, status TEXT NOT NULL DEFAULT 'mirrored',
                   created_at TEXT NOT NULL, superseded_by INTEGER,
                   UNIQUE(planned_session_id, status)
               )"""
        )

    init_db(database)

    with connect(database) as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 8
        columns = {row[1] for row in connection.execute("PRAGMA table_info(workout_prescriptions)")}
        assert {
            "action", "evidence_json", "rules_json", "engine_version",
            "optional_gate_json", "decision_trace_json", "recovery_runway_h",
            "protected_sessions_json",
        } <= columns
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'daily_checkins'"
        ).fetchone()
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'session_costs'"
        ).fetchone()
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'sync_runs'"
        ).fetchone()


def test_athlete_state_excludes_future_and_same_day_activity_data():
    database = _database()
    _activity(database, "2026-01-01", "cycling", 3600, load=40)
    _activity(database, "2026-01-05", "cycling", 7200, load=200)
    _activity(database, "2026-01-10", "running", 1800, load=30)

    state = athlete_state(date(2026, 1, 5), database)

    assert state.recent_training["7d"]["sessions"] == 1
    assert state.recent_training["7d"]["duration_seconds"] == 3600
    assert state.load["load_7d"] == 40


def test_performance_trend_excludes_future_observations():
    database = _database()
    for day in ("2025-12-08", "2025-12-10", "2025-12-12", "2025-12-14"):
        _activity(database, day, "cycling", 3600, efficiency=1.0)
    for day in ("2026-02-01", "2026-02-03", "2026-02-05", "2026-02-07"):
        _activity(database, day, "cycling", 3600, efficiency=1.0)
    _activity(database, "2026-03-02", "cycling", 3600, efficiency=100.0)

    trend = performance_trends(date(2026, 3, 1), database)["cycling"]

    assert trend.state == "STABLE"
    assert "100.00" not in " ".join(trend.evidence)


def test_classifier_uses_measured_intensity_and_exposes_hard_stress():
    database = _database()
    activity_id = _activity(
        database, "2026-01-01", "cycling", 3600,
        load=120, intensity_factor=0.96, name="Indoor ride",
    )

    result = classify_activity(activity_id, database)

    assert result.classification == "THRESHOLD"
    assert result.stress_level == "high"
    assert any("intensity factor" in reason for reason in result.reasons)


def test_missing_hr_power_and_subjective_feedback_do_not_prevent_keep():
    database = _database()
    _plan_session(database, "2026-01-08", "cycling", "Endurance ride", 3600, intensity="Z2")

    result = prescribe(date(2026, 1, 8), database, persist=False)

    assert result.action == "KEEP"
    assert result.prescribed["title"] == "Endurance ride"


def test_unusually_high_load_reduces_duration_without_increasing_intensity():
    database = _database()
    for offset in range(1, 7):
        _activity(database, f"2026-01-0{offset + 1}", "cycling", 3600, load=80, intensity_factor=0.70)
    _activity(database, "2025-12-15", "cycling", 1800, load=20, intensity_factor=0.60)
    _plan_session(database, "2026-01-08", "cycling", "Threshold 3x10", 5400, intensity="Threshold")

    result = prescribe(date(2026, 1, 8), database, persist=False)

    assert result.action == "REDUCE_DURATION"
    assert result.prescribed["duration_seconds"] < result.original["duration_seconds"]
    assert result.prescribed["intensity"] == result.original["intensity"]


def test_leg_strength_interference_reduces_quality_intensity():
    database = _database()
    strength_id = _activity(database, "2026-01-07", "strength", 3600, load=40)
    with connect(database) as connection:
        for number, exercise in enumerate(("goblet_squat", "leg_press", "leg_curl", "romanian_deadlift"), 1):
            connection.execute(
                """INSERT INTO strength_sets (activity_id, exercise_name, set_number, repetitions)
                   VALUES (?, ?, ?, 10)""",
                (strength_id, exercise, number),
            )
        connection.commit()
    _plan_session(database, "2026-01-08", "cycling", "Threshold ride", 4500, intensity="Threshold")

    result = prescribe(date(2026, 1, 8), database, persist=False)

    assert result.action == "REDUCE_INTENSITY"
    assert result.prescribed["targets"] == {}
    assert "leg_strength_interference" in result.rules_triggered


def test_high_stress_spacing_moves_quality_session_when_next_day_is_free():
    database = _database()
    _activity(
        database, "2026-01-07", "cycling", 3600,
        load=130, intensity_factor=0.96,
    )
    _plan_session(database, "2026-01-08", "cycling", "Threshold ride", 4500, intensity="Threshold")

    result = prescribe(date(2026, 1, 8), database, persist=False)

    assert result.action == "MOVE"
    assert result.prescribed["date"] == "2026-01-09"
    assert "high_stress_spacing" in result.rules_triggered


def test_missed_key_session_is_not_stacked_into_next_easy_day():
    database = _database()
    _plan_session(database, "2026-01-07", "cycling", "Threshold ride", 3600, intensity="Threshold", priority="A - Key")
    _plan_session(database, "2026-01-08", "cycling", "Easy endurance", 3600, intensity="Z2")

    result = prescribe(date(2026, 1, 8), database, persist=False)

    assert result.action == "KEEP"
    assert result.prescribed["title"] == "Easy endurance"


def test_high_subjective_feedback_reduces_intensity_but_missing_feedback_does_not():
    database = _database()
    _plan_session(database, "2026-01-08", "running", "Run intervals", 3600, intensity="Intervals")
    before = prescribe(date(2026, 1, 8), database, persist=False)
    save_checkin(date(2026, 1, 8), database, fatigue=4, leg_soreness=2)
    after = prescribe(date(2026, 1, 8), database, persist=False)

    assert before.action == "KEEP"
    assert after.action == "REDUCE_INTENSITY"


def test_maximum_optional_fatigue_and_soreness_can_select_rest():
    database = _database()
    _plan_session(database, "2026-01-08", "cycling", "Easy endurance", 3600, intensity="Z2")
    save_checkin(date(2026, 1, 8), database, fatigue=5, leg_soreness=5)

    result = prescribe(date(2026, 1, 8), database, persist=False)

    assert result.action == "REST"
    assert result.prescribed["duration_seconds"] == 0


def test_prescription_persistence_is_idempotent():
    database = _database()
    session_id = _plan_session(database, "2026-01-08", "cycling", "Endurance ride", 3600)

    first = prescribe(date(2026, 1, 8), database, persist=True)
    second = prescribe(date(2026, 1, 8), database, persist=True)

    assert first.prescription_id == second.prescription_id
    with connect(database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM workout_prescriptions WHERE planned_session_id = ? AND status = 'active'",
            (session_id,),
        ).fetchone()[0]
    assert count == 1


def test_execution_evaluation_detects_completed_power_intervals():
    database = _database()
    session_id = _plan_session(
        database, "2026-01-08", "cycling", "Threshold 3x15", 4500, intensity="Threshold",
        description="3x15 min at threshold",
    )
    with connect(database) as connection:
        connection.execute(
            """INSERT INTO planned_session_targets (
                   planned_session_id, target_type, raw_text, minimum_value, maximum_value, unit, confidence
               ) VALUES (?, 'power', '250-265 W', 250, 265, 'W', 'high')""",
            (session_id,),
        )
        connection.commit()
    activity_id = _activity(database, "2026-01-08", "cycling", 4500, load=110, intensity_factor=0.95)
    _match(database, session_id, activity_id)
    with connect(database) as connection:
        for number, (duration, power) in enumerate(((900, 257), (300, 120), (900, 261), (300, 115), (900, 258)), 1):
            connection.execute(
                """INSERT INTO activity_laps (
                       activity_id, lap_number, duration_seconds, distance_m, avg_hr, avg_cadence, avg_power_w
                   ) VALUES (?, ?, ?, 5000, 165, 90, ?)""",
                (activity_id, number, duration, power),
            )
        connection.commit()

    result = evaluate_activity(activity_id, database)

    assert result.execution_status == "successful"
    assert result.dimensions["interval_completion"] == "complete"
    assert len(result.intervals) == 3


def test_execution_does_not_compare_targets_when_linked_sports_differ():
    database = _database()
    session_id = _plan_session(database, "2026-01-08", "strength", "Gym 3x10", 3600)
    activity_id = _activity(
        database, "2026-01-08", "cycling", 3600, load=120, intensity_factor=0.95
    )
    _match(database, session_id, activity_id)

    result = evaluate_activity(activity_id, database)

    assert result.execution_status == "unknown"
    assert result.dimensions["session_compatibility"] == "different_sport"
    assert result.intervals == ()


def test_backtest_is_historical_and_reports_distribution():
    database = _database()
    _plan_session(database, "2026-01-05", "cycling", "Endurance", 3600)
    _plan_session(database, "2026-01-06", "running", "Easy run", 2400)
    _activity(database, "2026-01-20", "cycling", 7200, load=300, intensity_factor=1.0)

    result = backtest(date(2026, 1, 1), date(2026, 1, 10), database)

    assert result["planned_sessions_evaluated"] == 2
    assert result["decision_distribution"] == {"KEEP": 2}
    assert result["suspicious_behavior"] == []


def test_long_very_easy_ride_is_not_misclassified_as_high_cost():
    result = assess_session_cost(
        {"id": 1, "sport": "cycling", "moving_seconds": 3 * 3600, "avg_power_w": 60, "ftp_w": 260, "avg_hr": 92},
        {"athlete": {"aerobic_hr_upper_bpm": 155}, "coaching": {"cost_model": {}}},
    )

    assert result.cost_class == "recovery"
    assert result.systemic_cost == "low"
    assert result.duration_cost == "high"


def test_short_threshold_session_retains_high_intensity_cost():
    result = assess_session_cost(
        {"id": 2, "sport": "cycling", "moving_seconds": 35 * 60, "name": "Threshold intervals", "intensity_factor": 0.95},
        {"coaching": {"cost_model": {}}},
    )

    assert result.intensity_cost == "high"
    assert result.systemic_cost == "high"
    assert result.duration_cost == "low"


def test_strength_cost_distinguishes_upper_and_lower_body_work():
    config = {"coaching": {"cost_model": {"strength": {"meaningful_working_sets": 3, "high_working_sets": 5}}}}
    upper = assess_session_cost(
        {"id": 3, "sport": "strength", "moving_seconds": 3600}, config,
        {"movement_counts": {"upper": 5}, "working_sets": {"upper": 5}},
    )
    lower = assess_session_cost(
        {"id": 4, "sport": "strength", "moving_seconds": 3600}, config,
        {"movement_counts": {"lower": 5}, "working_sets": {"lower": 5}},
    )

    assert upper.muscle_load == "upper" and upper.muscular_cost == "moderate"
    assert lower.muscle_load == "lower" and lower.muscular_cost == "high"


def test_heat_is_context_and_reduces_confidence_without_erasing_cost():
    result = assess_session_cost(
        {"id": 5, "sport": "cycling", "moving_seconds": 3600, "name": "Threshold", "intensity_factor": 0.94, "avg_hr": 170, "max_temperature_c": 34},
        {"coaching": {"cost_model": {"heat_threshold_c": 30}}},
    )

    assert result.systemic_cost == "high"
    assert "heat" in result.context_flags
    assert result.confidence == "medium"


def test_optional_session_after_a_priority_key_is_conditional_with_rest_fallback():
    database = _database()
    _plan_session(database, "2026-01-10", "cycling", "Long ride", 10800, priority="A - Key")
    _plan_session(database, "2026-01-11", "running", "Optional easy run", 2400, priority="C - Optional")

    results = adjust_week(date(2026, 1, 8), database, as_of=date(2026, 1, 8), persist=False)
    sunday = next(item for item in results if item.date == date(2026, 1, 11))

    assert sunday.action == "CONDITIONAL"
    assert sunday.optional_gate is not None
    assert sunday.optional_gate.fallback["title"] == "Rest"


def test_week_adjustment_uses_only_activity_data_before_as_of_date():
    database = _database()
    _plan_session(database, "2026-01-08", "cycling", "Threshold ride", 4500, priority="A - Key")
    _activity(database, "2026-01-09", "cycling", 3600, load=200, intensity_factor=1.0, name="Future race")

    result = adjust_week(date(2026, 1, 8), database, as_of=date(2026, 1, 8), persist=False)[0]

    assert result.action == "KEEP"
    assert result.decision_trace["actual_context"] == []


def test_decision_trace_records_explicit_priority_and_protected_session():
    database = _database()
    _plan_session(database, "2026-01-08", "cycling", "Endurance", 3600, priority="B - Important")
    _plan_session(database, "2026-01-10", "cycling", "Long ride", 10800, priority="A - Key")

    result = adjust_week(date(2026, 1, 8), database, as_of=date(2026, 1, 8), persist=False)[0]

    assert result.decision_trace["planned_session"]["priority"] == "B"
    assert result.protected_sessions[0]["priority"] == "A"


def test_persisted_session_cost_is_idempotent():
    database = _database()
    activity_id = _activity(database, "2026-01-08", "cycling", 3600, load=100, intensity_factor=0.9)

    first = session_cost(activity_id, database, persist=True)
    second = session_cost(activity_id, database, persist=True)

    assert first == second
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session_costs WHERE activity_id=?", (activity_id,)).fetchone()[0] == 1


def test_supplied_week_scenario_makes_only_the_minimum_changes():
    database = _database()
    _plan_session(database, "2026-09-22", "cycling", "Threshold 3x15", 5400, intensity="Threshold", priority="A - Key")
    _plan_session(database, "2026-09-23", "running", "Easy run", 2400, intensity="Easy", priority="B - Important")
    _plan_session(database, "2026-09-24", "strength", "Supplementary gym", 3600, priority="C - Optional")
    _plan_session(database, "2026-09-25", "cycling", "Aerobic Z2", 5400, intensity="Z2", priority="B - Important")
    _plan_session(database, "2026-09-26", "cycling", "Long ride", 10800, intensity="Z2", priority="A - Key")
    _plan_session(database, "2026-09-27", "running", "Optional easy run", 2400, intensity="Easy", priority="C - Optional")
    _activity(database, "2026-09-22", "cycling", 5400, load=140, intensity_factor=0.95, name="Threshold 3x15")
    strength_id = _activity(database, "2026-09-23", "strength", 2760, load=35, name="Strength")
    ride_id = _activity(database, "2026-09-23", "cycling", 4740, load=15, intensity_factor=0.25, name="Family ride")
    with connect(database) as connection:
        for number, (exercise, load) in enumerate(
            (("machine_leg_press", 100), ("machine_leg_press", 150), ("machine_leg_press", 200),
             ("machine_leg_extension", 60), ("machine_leg_extension", 75), ("machine_leg_extension", 80)), 1
        ):
            connection.execute(
                "INSERT INTO strength_sets (activity_id, exercise_name, set_number, repetitions, load_value, load_unit) VALUES (?, ?, ?, 10, ?, 'kg')",
                (strength_id, exercise, number, load),
            )
        connection.execute("UPDATE activities SET avg_power_w=60, avg_hr=95 WHERE id=?", (ride_id,))
        connection.execute("UPDATE derived_activity_metrics SET ftp_w=265 WHERE activity_id=?", (ride_id,))
        connection.commit()

    results = {item.date: item for item in adjust_week(date(2026, 9, 24), database, as_of=date(2026, 9, 24), persist=False)}

    assert session_cost(ride_id, database, persist=False).systemic_cost == "low"
    assert session_cost(strength_id, database, persist=False).muscular_cost == "high"
    assert results[date(2026, 9, 24)].action == "REPLACE_WITH_EASY"
    assert results[date(2026, 9, 25)].action == "KEEP"
    assert results[date(2026, 9, 26)].action == "KEEP"
    assert results[date(2026, 9, 27)].action == "CONDITIONAL"


def _database() -> Path:
    root = Path("data/test-runs") / uuid4().hex
    root.mkdir(parents=True)
    database = root / "coaching.sqlite3"
    init_db(database)
    return database


def _activity(
    database: Path,
    day: str,
    sport: str,
    duration: float,
    *,
    load: float = 0,
    intensity_factor: float | None = None,
    efficiency: float | None = None,
    name: str = "Training",
) -> int:
    token = uuid4().hex
    with connect(database) as connection:
        cursor = connection.execute(
            """INSERT INTO imports (
                   source_filename, sha256, source_format, source, raw_path,
                   imported_at, status, activity_count
               ) VALUES (?, ?, 'fit', 'test', ?, ?, 'imported', 1)""",
            (f"{token}.fit", token, f"data/{token}.fit", f"{day}T12:00:00+00:00"),
        )
        import_id = int(cursor.lastrowid)
        cursor = connection.execute(
            """INSERT INTO activities (
                   stable_id, source_filename, raw_path, import_id, source_format, source,
                   quality_score, metadata_only, sport, name, started_at, elapsed_seconds,
                   moving_seconds, imported_at
               ) VALUES (?, ?, ?, ?, 'fit', 'test', 10, 0, ?, ?, ?, ?, ?, ?)""",
            (
                token, f"{token}.fit", f"data/{token}.fit", import_id, sport, name,
                f"{day}T12:00:00+00:00", duration, duration, f"{day}T13:00:00+00:00",
            ),
        )
        activity_id = int(cursor.lastrowid)
        connection.execute(
            """INSERT INTO derived_activity_metrics (
                   activity_id, metric_version, computed_at, intensity_factor,
                   selected_load, load_method, efficiency_factor,
                   hr_zones_json, power_zones_json, best_power_json
               ) VALUES (?, 1, ?, ?, ?, 'test', ?, '[]', '[]', '{}')""",
            (activity_id, f"{day}T14:00:00+00:00", intensity_factor, load, efficiency),
        )
        connection.commit()
    return activity_id


def _plan_session(
    database: Path,
    day: str,
    sport: str,
    title: str,
    duration: float,
    *,
    intensity: str | None = None,
    priority: str = "B - Important",
    description: str | None = None,
) -> int:
    with connect(database) as connection:
        plan = connection.execute("SELECT id FROM training_plans LIMIT 1").fetchone()
        if plan:
            plan_id = int(plan[0])
        else:
            cursor = connection.execute(
                """INSERT INTO training_plans (
                       name, source_path, source_filename, source_sha256, imported_at,
                       date_start, date_end, active
                   ) VALUES ('Test', 'test.xlsx', 'test.xlsx', 'hash', ?, '2026-01-01', '2026-12-31', 1)""",
                (f"{day}T00:00:00+00:00",),
            )
            plan_id = int(cursor.lastrowid)
        cursor = connection.execute(
            """INSERT INTO planned_sessions (
                   plan_id, planned_date, sport, session_type, title, description,
                   planned_duration_seconds, intensity, priority, source_sheet,
                   source_row, source_key, raw_source_text, active, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Training Calendar', 1, ?, '{}', 1, ?)""",
            (
                plan_id, day, sport, intensity, title, description, duration, intensity,
                priority, f"{day}:{sport}:{title}", f"{day}T00:00:00+00:00",
            ),
        )
        session_id = int(cursor.lastrowid)
        connection.commit()
    return session_id


def _match(database: Path, session_id: int, activity_id: int) -> None:
    with connect(database) as connection:
        connection.execute(
            """INSERT INTO planned_activity_matches (
                   planned_session_id, activity_id, match_score, match_method,
                   match_status, reason_json, is_selected, matched_at
               ) VALUES (?, ?, 100, 'test', 'matched', ?, 1, '2026-01-09T00:00:00+00:00')""",
            (session_id, activity_id, json.dumps({"signals": ["test"]})),
        )
        connection.commit()
