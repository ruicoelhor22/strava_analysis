from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openpyxl import Workbook

from endurance_lab.db import connect, init_db
from endurance_lab.metadata_activities import MetadataActivity, upsert_metadata_activity
from endurance_lab.plan_adherence import planned_sessions, weekly_adherence
from endurance_lab.plan_importer import import_training_plan
from endurance_lab.plan_matching import match_training_plan, set_manual_match
from endurance_lab.plan_parser import parse_workbook


def test_workbook_parser_extracts_dates_sessions_targets_and_raw_provenance():
    root = _test_dir()
    workbook = _write_plan(root / "coaching.xlsx")
    parsed = parse_workbook(workbook)

    assert parsed.date_start.isoformat() == "2026-09-21"
    assert parsed.date_end.isoformat() == "2026-09-27"
    assert len(parsed.weeks) == 1
    assert len(parsed.phases) == 1
    assert len(parsed.sessions) == 2
    cycling = parsed.sessions[0]
    assert cycling.sport == "cycling"
    assert cycling.planned_duration_seconds == 3600
    assert cycling.targets[0].target_type == "power"
    assert cycling.targets[0].minimum_value == 250
    assert cycling.targets[0].maximum_value == 270
    assert parsed.workbook_snapshot["sheets"]["Training Calendar"]["merged_ranges"] == ["A1:AI1"]


def test_plan_import_is_hash_idempotent_and_schema_is_version_six():
    root = _test_dir()
    database = root / "plan.sqlite3"
    workbook = _write_plan(root / "coaching.xlsx")

    first = import_training_plan(workbook, database)
    second = import_training_plan(workbook, database)

    assert not first.unchanged
    assert second.unchanged
    with connect(database) as connection:
        assert connection.execute("SELECT version FROM schema_meta").fetchone()[0] == 8
        assert connection.execute("SELECT COUNT(*) FROM training_plans").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM training_weeks").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM planned_sessions").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM workout_prescriptions").fetchone()[0] == 2
        columns = {row[1] for row in connection.execute("PRAGMA table_info(activities)")}
        assert "metadata_only" in columns


def test_matching_uses_exact_source_id_and_reports_ambiguous_candidates():
    root = _test_dir()
    database = root / "matching.sqlite3"
    workbook = _write_plan(root / "coaching.xlsx")
    import_training_plan(workbook, database)
    exact_id = upsert_metadata_activity(
        MetadataActivity(
            "123456789", datetime(2026, 9, 21, 8, tzinfo=timezone.utc),
            "Ride", "Threshold session", 3600,
        ),
        database,
    )
    upsert_metadata_activity(
        MetadataActivity(
            "223456789", datetime(2026, 9, 22, 7, tzinfo=timezone.utc),
            "WeightTraining", "Gym work", 2700,
        ),
        database,
    )
    upsert_metadata_activity(
        MetadataActivity(
            "323456789", datetime(2026, 9, 22, 18, tzinfo=timezone.utc),
            "WeightTraining", "Gym session", 2700,
        ),
        database,
    )

    counts = match_training_plan(database)
    assert counts["matched"] == 1
    assert counts["ambiguous"] == 1
    with connect(database) as connection:
        selected = connection.execute(
            "SELECT activity_id, match_method FROM planned_activity_matches WHERE is_selected = 1"
        ).fetchone()
    assert selected["activity_id"] == exact_id
    assert selected["match_method"] == "workbook_strava_id"


def test_automatic_matching_never_selects_a_different_sport():
    root = _test_dir()
    database = root / "different-sport.sqlite3"
    workbook = _write_plan(root / "coaching.xlsx")
    import_training_plan(workbook, database)
    upsert_metadata_activity(
        MetadataActivity(
            "923456789", datetime(2026, 9, 22, 7, tzinfo=timezone.utc),
            "Run", "Morning run", 2700,
        ),
        database,
    )

    match_training_plan(database)

    with connect(database) as connection:
        selected = connection.execute(
            """SELECT COUNT(*) FROM planned_activity_matches m
               JOIN planned_sessions p ON p.id = m.planned_session_id
               JOIN activities a ON a.id = m.activity_id
               WHERE m.is_selected = 1 AND p.sport != a.sport"""
        ).fetchone()[0]
    assert selected == 0


def test_manual_match_is_preserved_and_weekly_adherence_is_dimensioned():
    root = _test_dir()
    database = root / "manual.sqlite3"
    workbook = _write_plan(root / "coaching.xlsx")
    import_training_plan(workbook, database)
    activity_id = upsert_metadata_activity(
        MetadataActivity(
            "423456789", datetime(2026, 9, 22, 7, tzinfo=timezone.utc),
            "WeightTraining", "Manual gym", 2400,
        ),
        database,
    )
    with connect(database) as connection:
        session_id = connection.execute(
            "SELECT id FROM planned_sessions WHERE sport = 'strength'"
        ).fetchone()[0]
    set_manual_match(session_id, activity_id, "confirmed by athlete", database)
    match_training_plan(database)

    sessions = planned_sessions(database, today=datetime(2026, 9, 23).date())
    strength = next(item for item in sessions if item["sport"] == "strength")
    assert strength["match_status"] == "manual"
    assert strength["duration_adherence"] == "close"
    week = weekly_adherence(database, today=datetime(2026, 9, 23).date())[0]
    assert week["planned_sessions"] == 2
    assert week["completed_sessions"] == 1
    assert "strength" in week["actual_distance_by_sport"]


def test_metadata_only_activity_has_no_streams_and_is_not_downgraded():
    root = _test_dir()
    database = root / "metadata.sqlite3"
    init_db(database)
    activity_id = upsert_metadata_activity(
        MetadataActivity(
            "523456789", datetime(2026, 9, 20, tzinfo=timezone.utc),
            "Run", "Manual run", 1800, 5000,
        ),
        database,
    )
    with connect(database) as connection:
        row = connection.execute("SELECT * FROM activities WHERE id = ?", (activity_id,)).fetchone()
        streams = connection.execute(
            "SELECT COUNT(*) FROM activity_streams WHERE activity_id = ?", (activity_id,)
        ).fetchone()[0]
    assert row["metadata_only"] == 1
    assert row["source_format"] == "metadata"
    assert streams == 0


def _write_plan(path: Path) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Training Calendar"
    sheet.merge_cells("A1:AI1")
    sheet["A1"] = "Synthetic coaching calendar"
    headers = [
        "Date", "Day", "Week Start", "Phase", "Week Type", "Priority", "Sport",
        "Workout Code", "Planned Session", "Detailed Prescription",
        "Planned Duration (min)", "Planned Intensity", "Power Target", "HR Target",
        "Pace Target", "Gym Details", "Fuel / Hydration", "Session Purpose", "Status",
        "Actual Sport", "Strava Activity ID", "Actual Session", "Actual Duration (min)",
        "Distance (km)", "Elevation (m)", "Avg HR", "Max HR", "Avg Power",
        "Weighted Power", "RPE", "Estimated Session Load", "Completion %",
        "Coach Analysis / Notes", "Athlete Notes", "Constraints / Changes",
    ]
    for column, header in enumerate(headers, start=1):
        sheet.cell(3, column, header)
    values = {
        "Date": datetime(2026, 9, 21), "Day": "Mon", "Week Start": datetime(2026, 9, 21),
        "Phase": "Build", "Week Type": "Build", "Priority": "A - Key", "Sport": "Ride",
        "Workout Code": "BIKE-THR", "Planned Session": "Threshold 3x10",
        "Detailed Prescription": "3x10 min controlled threshold", "Planned Duration (min)": 60,
        "Planned Intensity": "Threshold", "Power Target": "250-270 W",
        "HR Target": "response only", "Strava Activity ID": 123456789,
    }
    for column, header in enumerate(headers, start=1):
        sheet.cell(4, column, values.get(header))
    values = {
        "Date": datetime(2026, 9, 22), "Day": "Tue", "Week Start": datetime(2026, 9, 21),
        "Phase": "Build", "Week Type": "Build", "Priority": "B - Important",
        "Sport": "WeightTraining", "Workout Code": "GYM-A", "Planned Session": "Gym A",
        "Detailed Prescription": "Controlled full-body strength", "Planned Duration (min)": 45,
        "Planned Intensity": "RPE 7",
    }
    for column, header in enumerate(headers, start=1):
        sheet.cell(5, column, values.get(header))
    for offset in range(2, 7):
        row = 4 + offset
        sheet.cell(row, 1, datetime(2026, 9, 21 + offset))
        sheet.cell(row, 3, datetime(2026, 9, 21))
        sheet.cell(row, 4, "Build")
        sheet.cell(row, 5, "Build")
    library = workbook.create_sheet("Workout Library")
    library.append(["Code", "Sport", "Session", "Structure"])
    library.append(["BIKE-THR", "Ride", "Threshold", "3x10 min"])
    workbook.save(path)
    return path


def _test_dir() -> Path:
    path = Path("data/test-runs") / uuid4().hex
    path.mkdir(parents=True)
    return path
