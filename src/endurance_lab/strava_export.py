from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Callable, Protocol

import requests

from endurance_lab.config import paths, project_path
from endurance_lab.fit import parse_fit
from endurance_lab.strava_manifest import (
    DEFAULT_COOKIES,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_MANIFEST,
    ManifestRow,
    load_manifest,
    resolve_private_path,
    save_manifest,
)
from endurance_lab.strength_json import parse_strength_json
from endurance_lab.tcx import parse_tcx


STRAVA_BASE_URL = "https://www.strava.com/activities"
SUPPORTED_SUFFIXES = {".fit": "fit", ".tcx": "tcx", ".json": "json"}
TRANSIENT_STATUSES = {408, 425, 500, 502, 503, 504}


class TransportError(Exception):
    """A retryable failure raised by an authenticated download transport."""


class ResponseLike(Protocol):
    status_code: int
    content: bytes
    headers: dict[str, str]


class SessionLike(Protocol):
    def get(self, url: str, **kwargs) -> ResponseLike: ...


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    source_format: str | None = None
    reason: str = ""
    authentication_page: bool = False


@dataclass(frozen=True)
class FetchResult:
    status: str
    status_code: int | None = None
    content: bytes = b""
    message: str = ""


@dataclass
class DownloadSummary:
    expected: int
    selected: int = 0
    already_present: int = 0
    downloaded: int = 0
    fallback_successes: int = 0
    failed: int = 0
    rate_limited: bool = False
    authentication_required: bool = False
    formats: Counter = field(default_factory=Counter)


def build_authenticated_session(cookies_path: str | Path | None = None) -> requests.Session:
    cookie_file = resolve_private_path(cookies_path, DEFAULT_COOKIES)
    if not cookie_file.is_file():
        raise FileNotFoundError(
            f"Authenticated Netscape cookies file not found: {cookie_file}. "
            "Log in to Strava yourself and export the browser cookies locally."
        )
    jar = MozillaCookieJar(str(cookie_file))
    try:
        jar.load(ignore_discard=True, ignore_expires=False)
    except Exception as exc:
        raise ValueError(f"Could not read Netscape cookies file {cookie_file}: {exc}") from exc
    if not any("strava.com" in cookie.domain.lower() for cookie in jar):
        raise ValueError(f"Cookies file contains no strava.com cookies: {cookie_file}")
    session = requests.Session()
    session.cookies.update(jar)
    session.headers.update(
        {
            "User-Agent": "EnduranceLab-LocalExport/0.1 (personal authenticated export)",
            "Accept": "application/octet-stream, application/xml, application/json, text/xml;q=0.9, */*;q=0.5",
        }
    )
    return session


def validate_activity_file(path: str | Path) -> ValidationResult:
    source = Path(path)
    if not source.is_file():
        return ValidationResult(False, reason="file does not exist")
    try:
        payload = source.read_bytes()
    except OSError as exc:
        return ValidationResult(False, reason=str(exc))
    detected = detect_payload_type(payload)
    if not detected.valid:
        return detected
    try:
        if detected.source_format == "fit":
            activities = parse_fit(source)
        elif detected.source_format == "tcx":
            activities = parse_tcx(source)
        else:
            activities = parse_strength_json(source)
    except Exception as exc:
        return ValidationResult(False, detected.source_format, f"malformed {detected.source_format.upper()}: {exc}")
    if not activities:
        return ValidationResult(False, detected.source_format, "file contains no activities")
    if detected.source_format == "fit" and any(
        activity.source_metadata.get("decoder_errors") for activity in activities
    ):
        errors = [
            str(error)
            for activity in activities
            for error in activity.source_metadata.get("decoder_errors", [])
        ]
        return ValidationResult(False, "fit", f"FIT decoder errors: {'; '.join(errors)}")
    return detected


def detect_payload_type(payload: bytes) -> ValidationResult:
    if not payload:
        return ValidationResult(False, reason="empty response")
    prefix = payload[:8192].lstrip(b"\xef\xbb\xbf\x00\t\r\n ").lower()
    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html") or b"<html" in prefix[:1024]:
        login = _looks_like_login_html(prefix)
        reason = "HTML login response rejected" if login else "HTML response is not an activity export"
        return ValidationResult(False, reason=reason, authentication_page=login)
    if len(payload) >= 12 and payload[8:12] == b".FIT" and payload[0] in range(12, 33):
        return ValidationResult(True, "fit")
    if prefix.startswith(b"<"):
        if b"trainingcenterdatabase" in prefix[:4096] or b"trainingcenterdatabase" in payload[-4096:].lower():
            return ValidationResult(True, "tcx")
        return ValidationResult(False, reason="XML response is not TCX")
    if prefix.startswith((b"{", b"[")):
        try:
            decoded = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return ValidationResult(False, reason=f"malformed JSON: {exc}")
        if not isinstance(decoded, dict):
            return ValidationResult(False, reason="JSON response is not an activity object")
        lowered = json.dumps(decoded, ensure_ascii=True).lower()
        if not any(key in decoded for key in ("start_time", "started_at", "start_date")):
            auth = any(token in lowered for token in ("authorization", "unauthorized", "login", "sign in"))
            return ValidationResult(False, reason="JSON response is not a supported strength activity", authentication_page=auth)
        return ValidationResult(True, "json")
    return ValidationResult(False, reason="unsupported binary response")


def _looks_like_login_html(prefix: bytes) -> bool:
    """Recognize a login document without flagging normal logged-in Strava HTML."""
    explicit_markers = (
        b"log in to strava",
        b"sign in to strava",
        b">log in<",
        b">sign in<",
        b"<title>log in",
        b"<title>login",
        b'action="/session"',
        b"action='/session'",
    )
    if any(marker in prefix for marker in explicit_markers):
        return True
    has_password = b'type="password"' in prefix or b"type='password'" in prefix
    has_email = any(
        marker in prefix
        for marker in (
            b'type="email"',
            b"type='email'",
            b'name="email"',
            b"name='email'",
        )
    )
    return has_password and has_email


class StravaDownloader:
    def __init__(
        self,
        session: SessionLike,
        *,
        delay_seconds: float = 2.0,
        max_retries: int = 2,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
        authentication_probe: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session = session
        self.delay_seconds = max(0.0, delay_seconds)
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.authentication_probe = authentication_probe
        self.sleep = sleep
        self.monotonic = monotonic
        self._last_request_at: float | None = None

    def run(
        self,
        manifest_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        *,
        dry_run: bool = False,
        limit: int | None = None,
        force: bool = False,
        emit: Callable[[str], None] = print,
    ) -> DownloadSummary:
        manifest = resolve_private_path(manifest_path, DEFAULT_MANIFEST)
        destination = resolve_private_path(output_dir, DEFAULT_DOWNLOAD_DIR)
        rows = load_manifest(manifest)
        if not rows:
            raise ValueError(f"Manifest is empty or missing: {manifest}")
        destination.mkdir(parents=True, exist_ok=True)
        summary = DownloadSummary(expected=len(rows))
        candidates: list[ManifestRow] = []

        for row in rows:
            existing = find_valid_download(row, destination)
            if existing is not None and not force:
                summary.already_present += 1
                summary.formats[existing[1]] += 1
                if not dry_run:
                    _mark_existing(row, existing[0], existing[1])
                continue
            candidates.append(row)

        selected = candidates[: max(0, limit)] if limit is not None else candidates
        summary.selected = len(selected)
        emit(f"Expected: {len(rows)}")
        emit(f"Already present: {summary.already_present}")
        emit(f"Requiring download: {len(candidates)}")
        emit(f"Selected this run: {len(selected)}")
        if dry_run:
            for row in selected:
                emit(f"would download {row.strava_id}: export_original -> export_tcx fallback")
            return summary

        save_manifest(rows, manifest)
        for position, row in enumerate(selected, start=1):
            emit(f"Downloading {position}/{len(selected)}: {row.strava_id}")
            try:
                outcome = self._download_one(row, destination)
            except Exception as exc:
                row.download_status = "failed"
                row.error = f"unexpected downloader error: {exc}"
                row.result_status = "internal_error"
                summary.failed += 1
                save_manifest(rows, manifest)
                continue

            if outcome == "downloaded":
                summary.downloaded += 1
                summary.formats[row.downloaded_format] += 1
                if "tcx fallback successful" in row.result_status:
                    summary.fallback_successes += 1
            elif outcome == "rate_limited":
                summary.failed += 1
                summary.rate_limited = True
            elif outcome == "authentication_required":
                summary.failed += 1
                summary.authentication_required = True
            else:
                summary.failed += 1
            save_manifest(rows, manifest)
            if summary.rate_limited or summary.authentication_required:
                emit("Stopping remaining requests to protect the account/session.")
                break
        return summary

    def _download_one(self, row: ManifestRow, destination: Path) -> str:
        notes: list[str] = []
        for endpoint in ("export_original", "export_tcx"):
            result = self._fetch(row, endpoint)
            label = "original" if endpoint == "export_original" else "tcx"
            if result.status == "rate_limited":
                _mark_failure(row, "rate_limited", result, notes + [f"{label}: HTTP 429"])
                return "rate_limited"
            if result.status == "authentication_required":
                if self._can_try_tcx_fallback(endpoint):
                    notes.append(
                        f"{label}: authentication-style response, but browser session is still authenticated"
                    )
                    continue
                _mark_failure(row, "authentication_required", result, notes + [f"{label}: authentication required"])
                return "authentication_required"
            if result.status != "response" or result.status_code != 200:
                notes.append(f"{label}: {result.message or f'HTTP {result.status_code}'}")
                continue

            temporary = destination / f".{row.strava_id}.{label}.part"
            try:
                temporary.write_bytes(result.content)
                validation = validate_activity_file(temporary)
                if not validation.valid:
                    notes.append(f"{label}: {validation.reason}")
                    if validation.authentication_page:
                        if self._can_try_tcx_fallback(endpoint):
                            notes.append(
                                f"{label}: browser session is still authenticated"
                            )
                            continue
                        _mark_failure(row, "authentication_required", result, notes)
                        return "authentication_required"
                    continue
                source_format = str(validation.source_format)
                filename = canonical_filename(row.strava_id, source_format)
                target = destination / filename
                temporary.replace(target)
                row.filename = filename
                row.downloaded_format = source_format
                row.download_status = "downloaded"
                row.downloaded_at = _now()
                row.sha256 = _sha256_bytes(result.content)
                row.http_status = str(result.status_code)
                row.result_status = (
                    "; ".join(notes + ["tcx fallback successful"])
                    if endpoint == "export_tcx" and notes
                    else f"{label}: downloaded"
                )
                row.error = ""
                return "downloaded"
            finally:
                if temporary.exists():
                    temporary.unlink()

        _mark_failure(row, "failed", result, notes)
        return "failed"

    def _can_try_tcx_fallback(self, endpoint: str) -> bool:
        """Allow one fallback when only the original export looks logged out.

        Strava can return a login-shaped HTTP 200 response for an individual
        original export even though the persistent browser session remains
        authenticated.  The TCX endpoint is safe to try in that case.  A
        missing, failed, or negative probe remains fail-closed.
        """
        if endpoint != "export_original" or self.authentication_probe is None:
            return False
        try:
            return bool(self.authentication_probe())
        except Exception:
            return False

    def _fetch(self, row: ManifestRow, endpoint: str) -> FetchResult:
        url = f"{STRAVA_BASE_URL}/{row.strava_id}/{endpoint}"
        for retry in range(self.max_retries + 1):
            self._throttle()
            row.attempts += 1
            row.last_attempt_at = _now()
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout_seconds,
                    allow_redirects=True,
                )
            except (requests.RequestException, TransportError) as exc:
                if retry < self.max_retries:
                    self.sleep(self.backoff_seconds * (2**retry))
                    continue
                return FetchResult("error", message=f"network error: {exc}")
            row.http_status = str(response.status_code)
            if response.status_code == 429:
                retry_after = str(response.headers.get("Retry-After", "")).strip()
                message = "rate limited; run stopped"
                if retry_after:
                    message += f" (Retry-After: {retry_after})"
                return FetchResult("rate_limited", 429, message=message)
            if response.status_code == 401:
                return FetchResult("authentication_required", 401, message="HTTP 401")
            if response.status_code in TRANSIENT_STATUSES and retry < self.max_retries:
                self.sleep(self.backoff_seconds * (2**retry))
                continue
            return FetchResult(
                "response",
                response.status_code,
                bytes(response.content),
                f"HTTP {response.status_code}",
            )
        return FetchResult("error", message="retry limit reached")

    def _throttle(self) -> None:
        now = self.monotonic()
        if self._last_request_at is not None:
            remaining = self.delay_seconds - (now - self._last_request_at)
            if remaining > 0:
                self.sleep(remaining)
        self._last_request_at = self.monotonic()


def canonical_filename(strava_id: str, source_format: str) -> str:
    if source_format not in SUPPORTED_SUFFIXES.values():
        raise ValueError(f"Unsupported activity format: {source_format}")
    return f"{strava_id}.{source_format}"


def find_valid_download(row: ManifestRow, directory: Path) -> tuple[Path, str] | None:
    candidates: list[Path] = []
    if row.filename:
        recorded = directory / Path(row.filename).name
        if _belongs_to_activity(recorded, row.strava_id):
            candidates.append(recorded)
    for path in sorted(directory.glob(f"{row.strava_id}*")):
        if _belongs_to_activity(path, row.strava_id) and path not in candidates:
            candidates.append(path)
    for path in candidates:
        validation = validate_activity_file(path)
        if not validation.valid:
            continue
        digest = _sha256_file(path)
        if row.sha256 and row.filename and Path(row.filename).name == path.name and digest != row.sha256:
            continue
        return path, str(validation.source_format)
    return None


def reconcile(
    manifest_path: str | Path | None = None,
    download_dir: str | Path | None = None,
    database: str | Path | None = None,
) -> dict[str, object]:
    manifest = resolve_private_path(manifest_path, DEFAULT_MANIFEST)
    directory = resolve_private_path(download_dir, DEFAULT_DOWNLOAD_DIR)
    rows = load_manifest(manifest)
    by_id = {row.strava_id: row for row in rows}
    files_by_id: defaultdict[str, list[tuple[Path, str]]] = defaultdict(list)
    invalid_files: list[str] = []
    if directory.exists():
        for file in sorted(path for path in directory.rglob("*") if path.is_file()):
            match = re.match(r"^(\d{7,20})(?:[._-].*)?\.(fit|tcx|json)$", file.name, re.IGNORECASE)
            if not match:
                continue
            validation = validate_activity_file(file)
            if validation.valid:
                files_by_id[match.group(1)].append((file, str(validation.source_format)))
            else:
                invalid_files.append(str(file))

    downloaded_ids = set(files_by_id)
    expected_ids = set(by_id)
    database_path = project_path(database) if database else paths().database
    imported_ids: set[str] = set()
    db_total = unidentified = import_failures = 0
    if database_path.exists():
        connection = sqlite3.connect(database_path)
        try:
            rows_db = connection.execute("SELECT source_activity_id FROM activities").fetchall()
            db_total = len(rows_db)
            imported_ids = {str(row[0]) for row in rows_db if row[0]}
            unidentified = sum(row[0] is None for row in rows_db)
            import_failures = int(
                connection.execute("SELECT COUNT(*) FROM imports WHERE status = 'failed'").fetchone()[0]
            )
        finally:
            connection.close()

    formats = Counter(fmt for values in files_by_id.values() for _, fmt in values)
    sports: dict[str, dict[str, int]] = {}
    for sport in sorted({row.sport or "unknown" for row in rows}):
        sport_ids = {row.strava_id for row in rows if (row.sport or "unknown") == sport}
        sports[sport] = {
            "expected": len(sport_ids),
            "downloaded": len(sport_ids & downloaded_ids),
            "imported": len(sport_ids & imported_ids),
        }
    return {
        "manifest": len(rows),
        "downloaded": len(expected_ids & downloaded_ids),
        "imported_matching_manifest": len(expected_ids & imported_ids),
        "database_activities": db_total,
        "missing_download_ids": sorted(expected_ids - downloaded_ids),
        "failed_download_ids": sorted(
            row.strava_id for row in rows if row.download_status in {"failed", "rate_limited", "authentication_required"}
        ),
        "downloaded_not_imported_ids": sorted((expected_ids & downloaded_ids) - imported_ids),
        "imported_not_in_manifest_ids": sorted(imported_ids - expected_ids),
        "database_records_without_source_id": unidentified,
        "database_import_failures": import_failures,
        "duplicate_download_ids": sorted(activity_id for activity_id, values in files_by_id.items() if len(values) > 1),
        "download_ids_not_in_manifest": sorted(downloaded_ids - expected_ids),
        "invalid_files": invalid_files,
        "format_distribution": dict(sorted(formats.items())),
        "sport_distribution": sports,
    }


def format_download_summary(summary: DownloadSummary) -> str:
    accounted = summary.already_present + summary.downloaded
    lines = [
        "STRAVA EXPORT SUMMARY",
        f"Expected activities: {summary.expected}",
        f"Selected this run: {summary.selected}",
        f"Downloaded this run: {summary.downloaded}",
        f"Already present: {summary.already_present}",
        f"TCX fallback successful: {summary.fallback_successes}",
        f"Completely failed: {summary.failed}",
        f"Total accounted for: {accounted}/{summary.expected}",
        f"Remaining or unresolved: {max(0, summary.expected - accounted)}",
    ]
    if summary.formats:
        lines.append("Formats: " + ", ".join(f"{key.upper()} {value}" for key, value in sorted(summary.formats.items())))
    if summary.rate_limited:
        lines.append("Stopped on HTTP 429; respect Retry-After before resuming.")
    if summary.authentication_required:
        lines.append("Stopped because the authenticated browser session is missing or expired.")
    return "\n".join(lines)


def format_reconciliation(report: dict[str, object]) -> str:
    lines = [
        "RECONCILIATION",
        f"Strava manifest: {report['manifest']}",
        f"Downloaded: {report['downloaded']}",
        f"Imported (matched): {report['imported_matching_manifest']}",
        f"Database activities: {report['database_activities']}",
        f"Missing downloads: {len(report['missing_download_ids'])}",
        f"Failed downloads: {len(report['failed_download_ids'])}",
        f"Downloaded but not imported: {len(report['downloaded_not_imported_ids'])}",
        f"Imported but not in manifest: {len(report['imported_not_in_manifest_ids'])}",
        f"DB records without source ID: {report['database_records_without_source_id']}",
        f"Import failures: {report['database_import_failures']}",
        f"Duplicate download IDs: {len(report['duplicate_download_ids'])}",
        "Formats: " + ", ".join(f"{key.upper()} {value}" for key, value in report["format_distribution"].items()),
    ]
    for label, key in (
        ("Missing download IDs", "missing_download_ids"),
        ("Failed download IDs", "failed_download_ids"),
        ("Downloaded not imported IDs", "downloaded_not_imported_ids"),
        ("Imported not in manifest IDs", "imported_not_in_manifest_ids"),
        ("Duplicate download IDs", "duplicate_download_ids"),
        ("Download IDs not in manifest", "download_ids_not_in_manifest"),
        ("Invalid files", "invalid_files"),
    ):
        if report[key]:
            lines.append(f"{label}: {', '.join(str(value) for value in report[key])}")
    for sport, counts in report["sport_distribution"].items():
        lines.append(
            f"{sport}: {counts['downloaded']}/{counts['expected']} downloaded, "
            f"{counts['imported']}/{counts['expected']} imported"
        )
    return "\n".join(lines)


def _mark_existing(row: ManifestRow, path: Path, source_format: str) -> None:
    row.filename = path.name
    row.downloaded_format = source_format
    row.download_status = "downloaded"
    row.sha256 = _sha256_file(path)
    row.result_status = "existing valid file"
    row.error = ""
    if not row.downloaded_at:
        row.downloaded_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _mark_failure(row: ManifestRow, status: str, result: FetchResult, notes: list[str]) -> None:
    row.download_status = status
    row.http_status = "" if result.status_code is None else str(result.status_code)
    row.result_status = "; ".join(notes) or result.message
    row.error = row.result_status or result.message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _belongs_to_activity(path: Path, strava_id: str) -> bool:
    return (
        path.suffix.lower() in SUPPORTED_SUFFIXES
        and re.match(rf"^{re.escape(strava_id)}(?:[._-])", path.name) is not None
    )
