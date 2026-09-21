from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from endurance_lab.importer import import_file
from endurance_lab.cli import main as cli_main
from endurance_lab.strava_export import (
    StravaDownloader,
    canonical_filename,
    detect_payload_type,
    reconcile,
    validate_activity_file,
)
from endurance_lab.strava_manifest import (
    ManifestRow,
    load_manifest,
    merge_manifest_source,
    save_manifest,
)
from tests.test_fit import _write_fit


@pytest.fixture
def tmp_path() -> Path:
    path = Path("data/test-runs") / uuid4().hex
    path.mkdir(parents=True)
    return path


@dataclass
class FakeResponse:
    status_code: int
    content: bytes = b""
    headers: dict[str, str] | None = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class QueueSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize("source_format", ["fit", "tcx", "json"])
def test_successful_supported_responses_are_detected_and_saved(tmp_path, source_format):
    payload = _payload(tmp_path, source_format, "123456789")
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession([FakeResponse(200, payload)])
    summary = _downloader(session).run(manifest, output)
    row = load_manifest(manifest)[0]
    assert summary.downloaded == 1
    assert row.download_status == "downloaded"
    assert row.filename == canonical_filename("123456789", source_format)
    assert row.sha256
    assert validate_activity_file(output / row.filename).valid


def test_original_failure_falls_back_to_tcx(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession([FakeResponse(404), FakeResponse(200, _tcx_bytes())])
    summary = _downloader(session).run(manifest, output)
    row = load_manifest(manifest)[0]
    assert summary.fallback_successes == 1
    assert row.downloaded_format == "tcx"
    assert row.attempts == 2
    assert "tcx fallback successful" in row.result_status


def test_html_login_and_unsupported_content_are_rejected(tmp_path):
    login = detect_payload_type(b"<!doctype html><html><body>Log in to Strava</body></html>")
    unsupported = detect_payload_type(b"\x00\x01not-an-activity")
    assert not login.valid and login.authentication_page
    assert not unsupported.valid

    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789"), ManifestRow("123456790")], manifest)
    session = QueueSession([FakeResponse(200, b"<html>Log in</html>")])
    summary = _downloader(session).run(manifest, tmp_path / "import")
    rows = load_manifest(manifest)
    assert summary.authentication_required
    assert rows[0].download_status == "authentication_required"
    assert rows[1].download_status == "pending"
    assert len(session.calls) == 1


def test_logged_in_activity_html_is_not_misclassified_as_login():
    page = detect_payload_type(
        b"<!doctype html><html class='logged-in old-login'>"
        b"<title>Morning Tennis | Workout | Strava</title>"
        b"<script>window.sessionStorage;</script></html>"
    )

    assert not page.valid
    assert not page.authentication_page
    assert page.reason == "HTML response is not an activity export"


def test_unavailable_activity_export_fails_individually_and_batch_continues(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    save_manifest([ManifestRow("123456789"), ManifestRow("123456790")], manifest)
    activity_page = FakeResponse(
        200,
        b"<!doctype html><html class='logged-in old-login'>"
        b"<title>Morning Tennis | Workout | Strava</title></html>",
    )
    session = QueueSession(
        [activity_page, activity_page, FakeResponse(200, _json_bytes())]
    )

    summary = _downloader(session).run(manifest, output)
    rows = load_manifest(manifest)

    assert summary.failed == 1
    assert summary.downloaded == 1
    assert not summary.authentication_required
    assert rows[0].download_status == "failed"
    assert rows[1].download_status == "downloaded"
    assert len(session.calls) == 3


def test_authenticated_session_uses_tcx_after_original_returns_login_html(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession(
        [FakeResponse(200, b"<html>Log in</html>"), FakeResponse(200, _tcx_bytes())]
    )
    downloader = StravaDownloader(
        session,
        delay_seconds=0,
        max_retries=0,
        authentication_probe=lambda: True,
        sleep=lambda _: None,
    )

    summary = downloader.run(manifest, output)
    row = load_manifest(manifest)[0]

    assert summary.downloaded == 1
    assert summary.fallback_successes == 1
    assert not summary.authentication_required
    assert row.downloaded_format == "tcx"
    assert "browser session is still authenticated" in row.result_status
    assert len(session.calls) == 2


def test_login_html_still_stops_when_authentication_probe_fails(tmp_path):
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession([FakeResponse(200, b"<html>Log in</html>")])
    downloader = StravaDownloader(
        session,
        delay_seconds=0,
        max_retries=0,
        authentication_probe=lambda: False,
        sleep=lambda _: None,
    )

    summary = downloader.run(manifest, tmp_path / "import")

    assert summary.authentication_required
    assert load_manifest(manifest)[0].download_status == "authentication_required"
    assert len(session.calls) == 1


def test_tcx_login_html_stops_even_when_session_probe_passes(tmp_path):
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789")], manifest)
    login = FakeResponse(200, b"<html>Log in</html>")
    session = QueueSession([login, login])
    downloader = StravaDownloader(
        session,
        delay_seconds=0,
        max_retries=0,
        authentication_probe=lambda: True,
        sleep=lambda _: None,
    )

    summary = downloader.run(manifest, tmp_path / "import")

    assert summary.authentication_required
    assert load_manifest(manifest)[0].download_status == "authentication_required"
    assert len(session.calls) == 2


def test_unsupported_and_malformed_downloads_fail_without_being_saved(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    save_manifest([ManifestRow("123456789")], manifest)
    malformed_fit = bytes([14, 0x20, 0, 0, 0, 0, 0, 0]) + b".FIT" + b"broken"
    session = QueueSession([FakeResponse(200, malformed_fit), FakeResponse(200, b"random binary")])
    summary = _downloader(session).run(manifest, output)
    row = load_manifest(manifest)[0]
    assert summary.failed == 1
    assert row.download_status == "failed"
    assert not list(output.glob("123456789.*"))
    assert "malformed FIT" in row.result_status


def test_valid_existing_file_is_skipped_and_manifest_is_repaired(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    output.mkdir()
    existing = output / "123456789.json"
    existing.write_bytes(_json_bytes("2026-01-01T07:00:00Z"))
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession([])
    summary = _downloader(session).run(manifest, output)
    row = load_manifest(manifest)[0]
    assert summary.already_present == 1
    assert summary.selected == 0
    assert session.calls == []
    assert row.filename == existing.name
    assert row.download_status == "downloaded"
    assert row.sha256


def test_resume_downloads_only_missing_rows(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    output.mkdir()
    (output / "123456789.json").write_bytes(_json_bytes("2026-01-01T07:00:00Z"))
    save_manifest([ManifestRow("123456789"), ManifestRow("123456790")], manifest)
    session = QueueSession([FakeResponse(200, _json_bytes("2026-01-02T07:00:00Z"))])
    summary = _downloader(session).run(manifest, output)
    assert summary.already_present == 1
    assert summary.downloaded == 1
    assert len(session.calls) == 1


def test_transient_failure_retries_then_succeeds(tmp_path):
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession([FakeResponse(503), FakeResponse(200, _json_bytes())])
    sleeps: list[float] = []
    downloader = StravaDownloader(
        session, delay_seconds=0, max_retries=2, backoff_seconds=1,
        sleep=sleeps.append,
    )
    summary = downloader.run(manifest, tmp_path / "import")
    assert summary.downloaded == 1
    assert load_manifest(manifest)[0].attempts == 2
    assert sleeps == [1]


def test_network_error_and_permanent_http_failures_are_recorded(tmp_path):
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession(
        [requests.ConnectionError("offline"), FakeResponse(404), FakeResponse(404)]
    )
    summary = StravaDownloader(
        session, delay_seconds=0, max_retries=1, backoff_seconds=0, sleep=lambda _: None
    ).run(manifest, tmp_path / "import")
    row = load_manifest(manifest)[0]
    assert summary.failed == 1
    assert row.download_status == "failed"
    assert row.attempts == 3
    assert "HTTP 404" in row.result_status


def test_http_429_stops_remaining_downloads(tmp_path):
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789"), ManifestRow("123456790")], manifest)
    session = QueueSession([FakeResponse(429, headers={"Retry-After": "120"})])
    summary = _downloader(session).run(manifest, tmp_path / "import")
    rows = load_manifest(manifest)
    assert summary.rate_limited
    assert rows[0].download_status == "rate_limited"
    assert rows[1].download_status == "pending"
    assert len(session.calls) == 1


def test_dry_run_requires_no_cookies_or_network_and_honors_limit(tmp_path):
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789"), ManifestRow("123456790")], manifest)
    lines: list[str] = []
    summary = StravaDownloader(None).run(  # type: ignore[arg-type]
        manifest, tmp_path / "import", dry_run=True, limit=1, emit=lines.append
    )
    assert summary.expected == 2
    assert summary.selected == 1
    assert any("would download 123456789" in line for line in lines)


def test_cli_dry_run_needs_no_cookie_file(tmp_path, capsys):
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789")], manifest)
    result = cli_main(
        [
            "strava-download", "--manifest", str(manifest), "--output",
            str(tmp_path / "import"), "--cookies", str(tmp_path / "missing.txt"),
            "--limit", "1", "--dry-run",
        ]
    )
    assert result == 0
    assert "would download 123456789" in capsys.readouterr().out


def test_existing_scan_does_not_confuse_prefix_activity_ids(tmp_path):
    output = tmp_path / "import"
    output.mkdir()
    (output / "1234567890.json").write_bytes(_json_bytes())
    manifest = tmp_path / "manifest.csv"
    save_manifest([ManifestRow("123456789")], manifest)
    session = QueueSession([FakeResponse(200, _json_bytes())])
    summary = _downloader(session).run(manifest, output)
    assert summary.already_present == 0
    assert summary.downloaded == 1
    assert (output / "123456789.json").exists()


def test_manifest_accepts_plain_ids_and_richer_csv(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("123456789\n123456790\n123456789\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    target, added, updated = merge_manifest_source(ids, manifest)
    assert target == manifest.resolve()
    assert (added, updated) == (2, 0)

    rich = tmp_path / "activities.csv"
    with rich.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["activity_id", "date", "type", "name"])
        writer.writeheader()
        writer.writerow(
            {"activity_id": "123456789", "date": "2026-01-01", "type": "Ride", "name": "Morning ride"}
        )
    _, added, updated = merge_manifest_source(rich, manifest)
    row = load_manifest(manifest)[0]
    assert (added, updated) == (0, 1)
    assert row.sport == "Ride"
    assert row.name == "Morning ride"


def test_reconciliation_reports_missing_unimported_unmatched_and_duplicates(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "import"
    output.mkdir()
    rows = [
        ManifestRow("123456789", sport="Ride"),
        ManifestRow("123456790", sport="Run"),
        ManifestRow("123456791", sport="Swim"),
    ]
    save_manifest(rows, manifest)
    first = output / "123456789.json"
    first.write_bytes(_json_bytes("2026-01-01T07:00:00Z"))
    (output / "123456789.original.json").write_bytes(_json_bytes("2026-01-01T07:00:00Z"))
    second = output / "123456790.json"
    second.write_bytes(_json_bytes("2026-01-02T07:00:00Z"))
    unlisted = output / "123456799.json"
    unlisted.write_bytes(_json_bytes("2026-01-03T07:00:00Z"))
    database = tmp_path / "reconcile.sqlite3"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(tmp_path / "raw"))
    assert import_file(first, database).status == "imported"
    assert import_file(unlisted, database).status == "imported"

    report = reconcile(manifest, output, database)
    assert report["manifest"] == 3
    assert report["downloaded"] == 2
    assert report["imported_matching_manifest"] == 1
    assert report["missing_download_ids"] == ["123456791"]
    assert report["downloaded_not_imported_ids"] == ["123456790"]
    assert report["imported_not_in_manifest_ids"] == ["123456799"]
    assert report["duplicate_download_ids"] == ["123456789"]
    assert report["sport_distribution"]["Ride"] == {
        "expected": 1, "downloaded": 1, "imported": 1
    }
    assert cli_main(
        [
            "strava-reconcile", "--manifest", str(manifest), "--downloads",
            str(output), "--database", str(database),
        ]
    ) == 1


def test_filename_generation_rejects_unsupported_formats():
    assert canonical_filename("123456789", "fit") == "123456789.fit"
    with pytest.raises(ValueError):
        canonical_filename("123456789", "gpx")


def _downloader(session) -> StravaDownloader:
    return StravaDownloader(
        session,
        delay_seconds=0,
        max_retries=0,
        backoff_seconds=0,
        sleep=lambda _: None,
    )


def _payload(tmp_path: Path, source_format: str, activity_id: str) -> bytes:
    if source_format == "fit":
        return _write_fit(tmp_path / f"{activity_id}.fit", seconds=30).read_bytes()
    if source_format == "tcx":
        return _tcx_bytes()
    return _json_bytes()


def _json_bytes(start: str = "2026-01-15T07:00:00Z") -> bytes:
    return json.dumps(
        {
            "start_time": start,
            "elapsed_time": 1800,
            "creator": {"name": "Hevy"},
            "sets": [{"exercise_type": "ROW", "repetitions": 10, "weight": 50}],
        }
    ).encode()


def _tcx_bytes() -> bytes:
    return b"""<TrainingCenterDatabase><Activities><Activity Sport="Running">
    <Id>2026-01-15T07:00:00Z</Id><Lap StartTime="2026-01-15T07:00:00Z">
    <TotalTimeSeconds>60</TotalTimeSeconds><DistanceMeters>150</DistanceMeters><Track>
    <Trackpoint><Time>2026-01-15T07:00:00Z</Time><DistanceMeters>0</DistanceMeters></Trackpoint>
    <Trackpoint><Time>2026-01-15T07:01:00Z</Time><DistanceMeters>150</DistanceMeters></Trackpoint>
    </Track></Lap></Activity></Activities></TrainingCenterDatabase>"""
