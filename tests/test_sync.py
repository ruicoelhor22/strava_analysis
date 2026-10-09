from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from endurance_lab.db import connect, init_db
from endurance_lab.strava_discovery import DiscoveryError, StravaDiscovery
from endurance_lab.strava_manifest import ManifestRow, load_manifest, save_manifest
from endurance_lab.sync_lock import SyncAlreadyRunning, SyncLock
import endurance_lab.sync_lock as sync_lock_module
from endurance_lab.sync_pipeline import SyncOptions, SyncPipeline, sync_status
import endurance_lab.sync_pipeline as sync_pipeline_module


@dataclass
class FakeResponse:
    status_code: int
    content: bytes = b""
    headers: dict[str, str] | None = None

    def __post_init__(self):
        self.headers = self.headers or {}


class DiscoveryTransport:
    def __init__(self, pages, exports=None):
        self.pages = list(pages)
        self.exports = list(exports or [])
        self.calls: list[str] = []
        self.context = object()

    def get(self, url, **kwargs):
        self.calls.append(url)
        if "training_activities" in url:
            value = self.pages.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        return self.exports.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_discovery_paginates_and_updates_manifest_without_duplicates():
    root, database, manifest = _paths()
    save_manifest([ManifestRow("10000001", name="Old title")], manifest)
    pages = [
        _page([_activity("10000003", "2026-09-24T08:00:00+0000"), _activity("10000002", "2026-09-23T08:00:00+0000")], 1, 3),
        _page([_activity("10000001", "2026-09-22T08:00:00+0000", name="New title")], 2, 3),
    ]

    result = StravaDiscovery(DiscoveryTransport(pages), delay_seconds=0).discover(
        date_from=date(2026, 9, 22), date_to=date(2026, 9, 24),
        manifest_path=manifest, database=database,
    )

    assert result.complete and result.pages_checked == 2
    assert (result.new, result.updated) == (2, 1)
    rows = load_manifest(manifest)
    assert len(rows) == 3
    assert next(row for row in rows if row.strava_id == "10000001").name == "New title"


def test_discovery_resumes_after_interrupted_page():
    root, database, manifest = _paths()
    first = DiscoveryTransport([
        _page([_activity("10000002", "2026-09-24T08:00:00+0000")], 1, 3),
        RuntimeError("offline"),
    ])
    with pytest.raises(DiscoveryError, match="offline"):
        StravaDiscovery(first, delay_seconds=0, max_retries=0).discover(
            date_from=date(2026, 9, 20), date_to=date(2026, 9, 24),
            manifest_path=manifest, database=database,
        )
    second = DiscoveryTransport([_page([_activity("10000001", "2026-09-20T08:00:00+0000")], 2, 3)])
    result = StravaDiscovery(second, delay_seconds=0).discover(
        date_from=date(2026, 9, 20), date_to=date(2026, 9, 24),
        manifest_path=manifest, database=database,
    )

    assert result.complete
    assert "page=2" in second.calls[0]
    assert {row.strava_id for row in load_manifest(manifest)} == {"10000001", "10000002"}


def test_discovery_classifies_rate_limit():
    root, database, manifest = _paths()
    with pytest.raises(DiscoveryError) as raised:
        StravaDiscovery(DiscoveryTransport([FakeResponse(429)]), delay_seconds=0).discover(
            date_from=date(2026, 9, 20), date_to=date(2026, 9, 24),
            manifest_path=manifest, database=database,
        )
    assert raised.value.code == "RATE_LIMITED"


def test_sync_lock_rejects_concurrent_process_and_recovers_stale_lock():
    root = Path("training_data") / "test-sync" / uuid4().hex
    lock_path = root / "sync.lock"
    with SyncLock(lock_path):
        with pytest.raises(SyncAlreadyRunning):
            with SyncLock(lock_path):
                pass
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text('{"pid": 99999999, "started_at": "2000-01-01T00:00:00+00:00"}', encoding="utf-8")
    with SyncLock(lock_path):
        assert lock_path.exists()
    assert not lock_path.exists()


def test_old_lock_with_live_owner_is_not_deleted(monkeypatch):
    root = Path("training_data") / "test-sync" / uuid4().hex
    lock_path = root / "sync.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text('{"pid": 1234, "started_at": "2000-01-01T00:00:00+00:00"}', encoding="utf-8")
    monkeypatch.setattr(sync_lock_module, "_pid_exists", lambda pid: pid == 1234)
    with pytest.raises(SyncAlreadyRunning):
        with SyncLock(lock_path):
            pass
    assert lock_path.exists()


def test_full_pipeline_metadata_fallback_and_second_run_are_idempotent(monkeypatch):
    root, database, manifest = _paths()
    downloads = root / "import"
    activity = _activity("10000001", "2026-09-24T08:00:00+0000", sport="WeightTraining")
    changed_activity = dict(activity, name="Renamed training")
    transports = [
        DiscoveryTransport([_page([activity], 1, 1)], [FakeResponse(404), FakeResponse(404)]),
        DiscoveryTransport([_page([changed_activity], 1, 1)], []),
    ]

    class Factory:
        def __init__(self, profile, headless):
            self.value = transports.pop(0)
        def __enter__(self):
            return self.value
        def __exit__(self, *args):
            return None

    auth = lambda context: type("Auth", (), {"authenticated": True, "reason": "test"})()
    pipeline = SyncPipeline(transport_factory=Factory, authentication_checker=auth)
    options = SyncOptions(
        date_from=date(2026, 9, 24), date_to=date(2026, 9, 24),
        manifest=manifest, downloads=downloads, database=database, delay_seconds=0,
        skip_calendar=True,
    )
    first = pipeline.run(options)
    second = pipeline.run(options)

    assert first.status == "success"
    assert first.discovery["new"] == 1
    assert first.imports["metadata_only"] == 1
    assert second.status == "success"
    assert second.discovery["new"] == 0
    assert second.discovery["updated"] == 1
    assert second.downloads["downloaded"] == 0
    assert second.imports["new"] == 0
    with connect(database) as connection:
        row = connection.execute("SELECT metadata_only, source, name FROM activities WHERE source_activity_id='10000001'").fetchone()
        assert tuple(row) == (1, "strava_metadata", "Renamed training")


def test_sync_dry_run_does_not_create_manifest_or_prescriptions():
    root, database, manifest = _paths()
    transport = DiscoveryTransport([_page([_activity("10000001", "2026-09-24T08:00:00+0000")], 1, 1)])

    class Factory:
        def __init__(self, profile, headless): pass
        def __enter__(self): return transport
        def __exit__(self, *args): return None

    auth = lambda context: type("Auth", (), {"authenticated": True, "reason": "test"})()
    result = SyncPipeline(transport_factory=Factory, authentication_checker=auth).run(SyncOptions(
        date_from=date(2026, 9, 24), date_to=date(2026, 9, 24), dry_run=True,
        manifest=manifest, downloads=root / "import", database=database, delay_seconds=0,
    ))

    assert result.status == "dry_run"
    assert not manifest.exists()
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(1) FROM workout_prescriptions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(1) FROM sync_runs").fetchone()[0] == 0


def test_authentication_failure_stops_before_discovery():
    root, database, manifest = _paths()
    transport = DiscoveryTransport([])

    class Factory:
        def __init__(self, profile, headless): pass
        def __enter__(self): return transport
        def __exit__(self, *args): return None

    auth = lambda context: type("Auth", (), {"authenticated": False, "reason": "expired"})()
    result = SyncPipeline(transport_factory=Factory, authentication_checker=auth).run(SyncOptions(
        date_from=date(2026, 9, 24), date_to=date(2026, 9, 24),
        manifest=manifest, downloads=root / "import", database=database,
    ))

    assert result.status == "stopped"
    assert result.error_code == "AUTH_REQUIRED"
    assert transport.calls == []


def test_coaching_failure_resumes_without_discovery_download_or_import(monkeypatch):
    root, database, manifest = _paths()
    activity = _activity("10000001", "2026-09-24T08:00:00+0000", sport="WeightTraining")
    transports = [DiscoveryTransport([_page([activity], 1, 1)], [FakeResponse(404), FakeResponse(404)])]

    class Factory:
        def __init__(self, profile, headless): self.value = transports.pop(0)
        def __enter__(self): return self.value
        def __exit__(self, *args): return None

    auth = lambda context: type("Auth", (), {"authenticated": True, "reason": "test"})()
    calls = {"count": 0}

    def flaky_adjust(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated coaching failure")
        return []

    monkeypatch.setattr(sync_pipeline_module, "adjust_week", flaky_adjust)
    pipeline = SyncPipeline(transport_factory=Factory, authentication_checker=auth)
    options = SyncOptions(
        date_from=date(2026, 9, 24), date_to=date(2026, 9, 24),
        manifest=manifest, downloads=root / "import", database=database, delay_seconds=0,
        skip_calendar=True,
    )
    first = pipeline.run(options)
    second = pipeline.run(options)

    assert first.status == "failed" and first.error_code == "COACHING_ERROR"
    assert second.status == "success"
    assert transports == []
    assert second.run_id == first.run_id
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(1) FROM activities").fetchone()[0] == 1


def test_abandoned_run_resumes_local_stages_without_strava(monkeypatch):
    root, database, manifest = _paths()
    save_manifest([], manifest)
    with connect(database) as connection:
        run_id = connection.execute(
            "INSERT INTO sync_runs(started_at,trigger,status,requested_from,requested_to) VALUES ('2026-09-24T08:00:00+00:00','test','running','2026-09-24','2026-09-24')"
        ).lastrowid
        for stage in ("authentication", "discovery", "downloads", "import", "quality", "analytics"):
            connection.execute(
                "INSERT INTO sync_run_stages(sync_run_id,stage,status,started_at,finished_at) VALUES (?,?, 'success','2026-09-24T08:00:00+00:00','2026-09-24T08:01:00+00:00')",
                (run_id, stage),
            )
        connection.commit()
    assert sync_status(database, manifest, root / "import")["last_run"]["status"] == "interrupted"

    def no_transport(*args, **kwargs):
        raise AssertionError("Resuming local stages must not contact Strava")

    pipeline = SyncPipeline(transport_factory=no_transport)
    result = pipeline.run(SyncOptions(
        date_from=date(2026, 9, 24), date_to=date(2026, 9, 24),
        manifest=manifest, downloads=root / "import", database=database,
        skip_calendar=True,
    ))
    assert result.status == "success" and result.run_id == run_id
    with connect(database) as connection:
        row = connection.execute("SELECT status,finished_at FROM sync_runs WHERE id=?", (run_id,)).fetchone()
        assert row["status"] == "success" and row["finished_at"]
        assert connection.execute("SELECT status FROM sync_run_stages WHERE sync_run_id=? AND stage='calendar'", (run_id,)).fetchone()[0] == "skipped"


def test_abandoned_run_with_all_stages_completed_is_finalized_from_checkpoints():
    _root, database, _manifest = _paths()
    with connect(database) as connection:
        run_id = connection.execute(
            "INSERT INTO sync_runs(started_at,trigger,status,requested_from,requested_to) VALUES ('2026-09-24T08:00:00+00:00','test','running','2026-09-24','2026-09-24')"
        ).lastrowid
        for stage in ("authentication", "discovery", "downloads", "import", "quality",
                      "analytics", "reconciliation", "coaching", "calendar"):
            checkpoint = '{"new": 2}' if stage == "discovery" else '{}'
            connection.execute(
                """INSERT INTO sync_run_stages(sync_run_id,stage,status,started_at,finished_at,checkpoint_json)
                   VALUES (?,?,'success','2026-09-24T08:00:00+00:00','2026-09-24T08:01:00+00:00',?)""",
                (run_id, stage, checkpoint),
            )
        connection.commit()
    assert sync_pipeline_module._recover_abandoned_runs(database) == [run_id]
    with connect(database) as connection:
        row = connection.execute("SELECT status,discovered_count,summary_json FROM sync_runs WHERE id=?", (run_id,)).fetchone()
        assert row["status"] == "success" and row["discovered_count"] == 2
        assert json.loads(row["summary_json"])["discovery"]["new"] == 2


def test_sync_finalizes_when_interrupted_after_analytics(monkeypatch):
    root, database, manifest = _paths()
    transport = DiscoveryTransport(
        [_page([_activity("10000001", "2026-09-24T08:00:00+0000")], 1, 1)],
        [FakeResponse(404), FakeResponse(404)],
    )

    class Factory:
        def __init__(self, profile, headless): pass
        def __enter__(self): return transport
        def __exit__(self, *args): return None

    auth = lambda context: type("Auth", (), {"authenticated": True, "reason": "test"})()
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(sync_pipeline_module, "reconcile", interrupted)
    with pytest.raises(KeyboardInterrupt):
        SyncPipeline(transport_factory=Factory, authentication_checker=auth).run(SyncOptions(
            date_from=date(2026, 9, 24), date_to=date(2026, 9, 24),
            manifest=manifest, downloads=root / "import", database=database,
            skip_calendar=True,
        ))
    with connect(database) as connection:
        row = connection.execute("SELECT status,finished_at FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
        assert row["status"] == "interrupted" and row["finished_at"]


def test_calendar_failure_is_persisted_and_finalizes_partial(monkeypatch):
    root, database, manifest = _paths()
    transport = DiscoveryTransport(
        [_page([_activity("10000001", "2026-09-24T08:00:00+0000")], 1, 1)],
        [FakeResponse(404), FakeResponse(404)],
    )

    class Factory:
        def __init__(self, profile, headless): pass
        def __enter__(self): return transport
        def __exit__(self, *args): return None

    auth = lambda context: type("Auth", (), {"authenticated": True, "reason": "test"})()
    monkeypatch.setattr(sync_pipeline_module, "calendar_enabled", lambda: True)
    monkeypatch.setattr(sync_pipeline_module, "sync_google_calendar", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    result = SyncPipeline(transport_factory=Factory, authentication_checker=auth).run(SyncOptions(
        date_from=date(2026, 9, 24), date_to=date(2026, 9, 24),
        manifest=manifest, downloads=root / "import", database=database, delay_seconds=0,
    ))
    assert result.status == "partial" and result.calendar["error_code"] == "CALENDAR_ERROR"
    with connect(database) as connection:
        assert connection.execute("SELECT status FROM sync_run_stages WHERE stage='calendar'").fetchone()[0] == "failed"
        run = connection.execute("SELECT status,finished_at FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
        assert run["status"] == "partial" and run["finished_at"]


def test_verified_summary_counts_volume_even_when_detail_export_is_html():
    root,database,manifest=_paths()
    from endurance_lab.strava_manifest import save_manifest
    row=ManifestRow('10000009',activity_date='2026-10-04T09:30:00+00:00',sport='Padel',name='Padel 1h',duration_seconds='3600',distance_m='2000',discovery_source='strava_training_activities',download_status='failed',result_status='original: HTML response is not an activity export; tcx: HTML response is not an activity export')
    save_manifest([row],manifest)
    ids,summary=sync_pipeline_module._import_and_fallback(manifest,root/'import',{'10000009'},database)
    assert len(ids)==1 and summary['metadata_only']==1
    with connect(database) as connection:
        actual=connection.execute("SELECT metadata_only,elapsed_seconds FROM activities WHERE source_activity_id='10000009'").fetchone()
        assert tuple(actual)==(1,3600)
    stored=load_manifest(manifest)[0]
    assert stored.download_status=='failed' and stored.import_status=='imported'
    sync_pipeline_module._import_and_fallback(manifest,root/'import',{'10000009'},database)
    with connect(database) as connection:
        assert connection.execute('SELECT count(*) FROM activities').fetchone()[0]==1


def test_per_event_calendar_errors_are_not_reported_as_success(monkeypatch):
    from types import SimpleNamespace
    root,database,_=_paths()
    summary=dict(created=0,updated=0,unchanged=0,errors=1)
    monkeypatch.setattr(sync_pipeline_module,'calendar_enabled',lambda:True)
    monkeypatch.setattr(sync_pipeline_module,'sync_google_calendar',lambda *a,**k:SimpleNamespace(to_dict=lambda:summary))
    monkeypatch.setattr(sync_pipeline_module,'sync_activity_calendar',lambda *a,**k:SimpleNamespace(to_dict=lambda:{**summary,'errors':0}))
    result=sync_pipeline_module.SyncResult()
    calendar=sync_pipeline_module._update_calendar(result,SyncOptions(database=database))
    assert calendar['error_code']=='CALENDAR_ERROR' and result.warnings


def test_cli_reports_calendar_auth_failure_and_recovery_command():
    from endurance_lab.cli import _format_sync
    output=_format_sync({'status':'partial','calendar':{'error_code':'GOOGLE_AUTH_REQUIRED','error':'Token expired'},'warnings':['Authorization needs renewal']})
    assert 'Calendar FAILED' in output and 'calendar-login' in output
    assert 'Authorization needs renewal' in output


def _paths():
    root = Path("data/test-runs") / uuid4().hex
    root.mkdir(parents=True)
    database = root / "sync.sqlite3"
    manifest = root / "manifest.csv"
    init_db(database)
    return root, database, manifest


def _activity(activity_id, started_at, name="Training", sport="Ride"):
    return {
        "id_str": activity_id, "name": name, "sport_type": sport,
        "start_time": started_at, "moving_time_raw": 1800,
        "elapsed_time_raw": 1900, "distance_raw": 10000.0,
        "elevation_gain_raw": 100.0,
    }


def _page(models, page, total):
    return FakeResponse(200, json.dumps({
        "models": models, "page": page, "perPage": 2, "total": total,
    }).encode("utf-8"), {"content-type": "application/json"})
