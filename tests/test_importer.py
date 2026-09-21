from pathlib import Path
from uuid import uuid4

from endurance_lab.analytics import analyze_database
from endurance_lab.db import connect, init_db
from endurance_lab.importer import import_file


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_ride.tcx"


def test_import_is_idempotent_and_preserves_normalized_rows(monkeypatch):
    tmp_path = _test_dir()
    database = tmp_path / "test.sqlite3"
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(raw_dir))
    first = import_file(FIXTURE, database)
    second = import_file(FIXTURE, database)
    assert first.status == "imported"
    assert first.imported_activities == 1
    assert second.status == "duplicate"
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM activity_laps").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM activity_streams").fetchone()[0] == 4
    assert len(list((raw_dir / "tcx").glob("*.tcx"))) == 1


def test_analytics_can_be_recalculated_without_changing_raw_rows(monkeypatch):
    tmp_path = _test_dir()
    database = tmp_path / "test.sqlite3"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(tmp_path / "raw"))
    import_file(FIXTURE, database)
    assert analyze_database(database) == 1
    assert analyze_database(database) == 1
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM derived_activity_metrics").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM daily_training_load").fetchone()[0] == 1
        metric = connection.execute("SELECT load_method FROM derived_activity_metrics").fetchone()[0]
    assert metric in {"estimated_power_tss", "estimated_edwards_hr", "duration_only_estimate"}


def test_schema_has_query_indexes():
    tmp_path = _test_dir()
    database = tmp_path / "test.sqlite3"
    init_db(database)
    with connect(database) as connection:
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'index'")}
    assert "idx_activities_started_sport" in names
    assert "idx_streams_activity_elapsed" in names


def _test_dir() -> Path:
    path = Path("data/test-runs") / uuid4().hex
    path.mkdir(parents=True)
    return path
