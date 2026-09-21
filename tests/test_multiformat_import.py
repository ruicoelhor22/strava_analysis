from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from endurance_lab.db import connect
from endurance_lab.importer import import_file, import_path
from endurance_lab.quality import ingestion_quality_report
from endurance_lab.strength_json import parse_strength_json
from tests.test_fit import _write_fit


def test_strength_json_parses_observed_hevy_shape():
    source = _write_strength_json(_test_dir() / "123456789.json")
    strength = parse_strength_json(source)[0]
    assert strength.source_activity_id == "123456789"
    assert strength.sport == "strength"
    assert strength.elapsed_seconds == 1800
    assert strength.calories == 0
    assert strength.device == "Hevy"
    assert len(strength.strength_sets) == 2
    assert strength.strength_sets[0].exercise_name == "LAT_PULLDOWN"
    assert strength.strength_sets[0].repetitions == 10
    assert strength.strength_sets[0].load_value == 70
    assert strength.strength_sets[0].load_unit is None


def test_strength_json_allows_missing_optional_fields():
    root = _test_dir()
    source = root / "minimal.json"
    source.write_text(json.dumps({"start_time": "2026-01-15T07:00:00Z"}), encoding="utf-8")
    strength = parse_strength_json(source)[0]
    assert strength.sport == "strength"
    assert strength.elapsed_seconds is None
    assert strength.strength_sets == []


def test_mixed_import_continues_after_malformed_and_unsupported(monkeypatch):
    root = _test_dir()
    _write_strength_json(root / "good.json")
    (root / "bad.json").write_text("{broken", encoding="utf-8")
    (root / "broken.fit").write_bytes(b"not-fit")
    (root / "notes.txt").write_text("unsupported", encoding="utf-8")
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    results = import_path(root, root / "mixed.sqlite3")
    statuses = {result.file.name: result.status for result in results if result.file.parent == root.resolve()}
    assert statuses == {
        "bad.json": "failed",
        "broken.fit": "failed",
        "good.json": "imported",
        "notes.txt": "unsupported",
    }


def test_tcx_then_richer_fit_upgrades_without_duplicate(monkeypatch):
    root = _test_dir()
    database = root / "upgrade.sqlite3"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    tcx = root / "987654321.tcx"
    _write_sparse_tcx(tcx)
    fit = _write_fit(root / "987654321.fit", seconds=120)
    first = import_file(tcx, database)
    second = import_file(fit, database)
    assert first.status == "imported"
    assert second.status == "upgraded"
    assert second.upgraded_activities == 1
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 1
        row = connection.execute("SELECT source_format, avg_power_w FROM activities").fetchone()
        assert row["source_format"] == "fit"
        assert row["avg_power_w"] == 200
        assert connection.execute("SELECT COUNT(*) FROM activity_sources").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM activity_streams").fetchone()[0] == 121


def test_strength_sets_are_stored_without_invented_units(monkeypatch):
    root = _test_dir()
    source = _write_strength_json(root / "strength.json")
    database = root / "strength.sqlite3"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    assert import_file(source, database).status == "imported"
    with connect(database) as connection:
        row = connection.execute("SELECT * FROM strength_sets ORDER BY id LIMIT 1").fetchone()
    assert row["exercise_name"] == "LAT_PULLDOWN"
    assert row["load_value"] == 70
    assert row["load_unit"] is None


def test_quality_report_summarizes_formats_coverage_and_flags(monkeypatch):
    root = _test_dir()
    database = root / "quality.sqlite3"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    import_file(_write_fit(root / "987654324.fit", seconds=120), database)
    import_file(_write_strength_json(root / "strength.json"), database)
    report = ingestion_quality_report(database)
    assert report["total_activities"] == 2
    assert report["activities_by_sport"] == {"cycling": 1, "strength": 1}
    assert report["activities_by_source_format"] == {"fit": 1, "json": 1}
    assert report["activities_containing"]["heart_rate"] == 1
    assert report["activities_containing"]["gps"] == 1
    assert report["activities_containing"]["power"] == 1


def _write_strength_json(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "start_time": "2026-01-15T07:00:00Z",
                "utc_offset": 0,
                "elapsed_time": 1800,
                "active_time": 1800,
                "total_calories": 0,
                "creator": {"name": "Hevy"},
                "sets": [
                    {"exercise_type": "LAT_PULLDOWN", "repetitions": 10, "weight": 70, "start_time": "2026-01-15T07:05:00Z"},
                    {"exercise_type": "LAT_PULLDOWN", "repetitions": 8, "weight": 70, "start_time": "2026-01-15T07:07:00Z"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_sparse_tcx(path: Path) -> None:
    path.write_text(
        """<TrainingCenterDatabase><Activities><Activity Sport="Biking">
        <Id>2026-01-15T07:00:00Z</Id><Lap StartTime="2026-01-15T07:00:00Z">
        <TotalTimeSeconds>120</TotalTimeSeconds><DistanceMeters>1200</DistanceMeters><Track>
        <Trackpoint><Time>2026-01-15T07:00:00Z</Time><DistanceMeters>0</DistanceMeters></Trackpoint>
        <Trackpoint><Time>2026-01-15T07:02:00Z</Time><DistanceMeters>1200</DistanceMeters></Trackpoint>
        </Track></Lap></Activity></Activities></TrainingCenterDatabase>""",
        encoding="utf-8",
    )


def _test_dir() -> Path:
    path = Path("data/test-runs") / uuid4().hex
    path.mkdir(parents=True)
    return path
