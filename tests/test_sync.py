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
from endurance_lab.sync_pipeline import SyncOptions, SyncPipeline
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
