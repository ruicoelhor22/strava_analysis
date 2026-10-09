from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from endurance_lab.archive import archive_status, failure_reason, reconcile_archive
from endurance_lab.db import connect, init_db
from endurance_lab.importer import import_file
from endurance_lab.metadata_activities import MetadataActivity, upsert_metadata_activity
from endurance_lab.strava_manifest import ManifestRow, load_manifest, save_manifest
from tests.test_fit import _write_fit


def _paths():
    root = Path("data/test-runs") / uuid4().hex
    root.mkdir(parents=True)
    database = root / "archive.sqlite3"
    manifest = root / "manifest.csv"
    init_db(database)
    return root, database, manifest


def test_ambiguous_html_is_not_non_exportable_but_activity_is_known():
    root, database, manifest = _paths()
    row = ManifestRow(
        "10000001", activity_date="2026-09-24T08:00:00+00:00",
        sport="WeightTraining", name="Gym", duration_seconds="1800",
        download_status="failed", http_status="200",
        result_status="original: HTML response is not an activity export; tcx: HTML response is not an activity export",
    )
    save_manifest([row], manifest)
    assert failure_reason(row) == "ambiguous_html"
    dry = reconcile_archive(database, manifest, root / "import", dry_run=True)
    assert dry["actions"]["metadata_candidates"] == 1
    assert archive_status(database, manifest, root / "import")["canonical_activities"] == 0
    real = reconcile_archive(database, manifest, root / "import")
    assert real["after"]["provisional_metadata_only"] == 1
    assert real["after"]["detail_recovery_candidates"] == 1
    assert real["after"]["missing_activity_records"] == 0
    assert real["after"]["status"] == "KNOWN"
    assert real["after"]["detailed_data_coverage_percent"] == 0.0
    assert load_manifest(manifest)[0].download_status == "failed"
    with connect(database) as connection:
        activity = connection.execute("SELECT metadata_only,sport FROM activities").fetchone()
    assert tuple(activity) == (1, "strength")
    again = reconcile_archive(database, manifest, root / "import")
    assert again["after"]["canonical_activities"] == 1


def test_two_missing_endpoints_resolve_as_metadata_only():
    root, database, manifest = _paths()
    save_manifest([ManifestRow(
        "10000002", activity_date="2026-09-24T08:00:00+00:00",
        sport="Ride", download_status="failed",
        result_status="original: HTTP 404; tcx: HTTP 404",
    )], manifest)
    result = reconcile_archive(database, manifest, root / "import")
    assert result["after"]["resolved_metadata_only"] == 1
    assert result["after"]["detail_recovery_candidates"] == 0
    assert result["after"]["status"] == "KNOWN"


def test_local_database_fills_manifest_without_reimport():
    root, database, manifest = _paths()
    save_manifest([ManifestRow("10000003", download_status="downloaded")], manifest)
    with connect(database) as connection:
        import_id = connection.execute(
            """INSERT INTO imports(source_filename,sha256,source_format,raw_path,imported_at,status)
               VALUES ('10000003.fit','archive-test-10000003','fit','private','2026-09-24T08:00:00+00:00','imported')"""
        ).lastrowid
        connection.execute(
            """INSERT INTO activities(stable_id,source_activity_id,source_filename,raw_path,source_format,source,
                   import_id,sport,started_at,imported_at) VALUES ('strava:10000003','10000003','10000003.fit','private',
                   'fit','manual_file',?,'cycling','2026-09-23T08:00:00+00:00','2026-09-24T08:00:00+00:00')""",
            (import_id,),
        )
        connection.commit()
    result = reconcile_archive(database, manifest, root / "import")
    row = load_manifest(manifest)[0]
    assert row.activity_date == "2026-09-23T08:00:00+00:00"
    assert row.import_status == "imported" and row.db_activity_id
    assert result["after"]["resolved_detailed"] == 1


def test_metadata_reconciliation_links_unique_detailed_source_instead_of_duplicating():
    root, database, manifest = _paths()
    save_manifest([ManifestRow(
        "10000004", activity_date="2026-09-24T08:00:00+00:00", sport="Ride",
        duration_seconds="1800", distance_m="10000", download_status="failed",
        result_status="original: HTML response is not an activity export; tcx: HTML response is not an activity export",
    )], manifest)
    with connect(database) as connection:
        import_id = connection.execute(
            """INSERT INTO imports(source_filename,sha256,source_format,raw_path,imported_at,status)
               VALUES ('local.fit','archive-test-local','fit','private','2026-09-24T08:00:00+00:00','imported')"""
        ).lastrowid
        connection.execute(
            """INSERT INTO activities(stable_id,source_filename,raw_path,source_format,source,
                   import_id,sport,started_at,elapsed_seconds,distance_m,imported_at)
               VALUES ('local:unique','local.fit','private','fit','manual_file',?,'cycling',
                       '2026-09-24T08:00:00+00:00',1800,10000,'2026-09-24T08:00:00+00:00')""",
            (import_id,),
        )
        connection.commit()
    result = reconcile_archive(database, manifest, root / "import")
    assert result["actions"]["linked_cross_source"] == 1
    assert result["after"]["canonical_activities"] == 1
    assert result["after"]["resolved_detailed"] == 1
    with connect(database) as connection:
        assert connection.execute("SELECT source_activity_id FROM activities").fetchone()[0] == "10000004"


def test_detailed_export_clears_metadata_only_flag(monkeypatch):
    root, database, _manifest = _paths()
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    activity_id = upsert_metadata_activity(MetadataActivity(
        "10000005", datetime(2026, 1, 15, tzinfo=timezone.utc), "cycling", duration_seconds=120,
    ), database)
    source = _write_fit(root / "10000005.fit", seconds=120)
    result = import_file(source, database)
    assert result.status == "upgraded"
    with connect(database) as connection:
        row = connection.execute("SELECT id,metadata_only,source_format FROM activities").fetchone()
    assert tuple(row) == (activity_id, 0, "fit")
