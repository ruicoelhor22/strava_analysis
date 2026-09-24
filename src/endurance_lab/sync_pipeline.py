from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from endurance_lab.analytics import analyze_database
from endurance_lab.config import load_athlete_config, project_path
from endurance_lab.db import connect, init_db
from endurance_lab.google_calendar import (
    GoogleCalendarError, calendar_enabled, sync_activity_calendar, sync_google_calendar,
)
from endurance_lab.importer import import_file
from endurance_lab.metadata_activities import MetadataActivity, upsert_metadata_activity
from endurance_lab.plan_matching import match_training_plan
from endurance_lab.quality import ingestion_quality_report
from endurance_lab.session_cost import session_cost
from endurance_lab.strava_auth import PlaywrightTransport, check_authentication
from endurance_lab.strava_discovery import DiscoveryError, StravaDiscovery, default_discovery_window
from endurance_lab.strava_export import StravaDownloader, find_valid_download, reconcile
from endurance_lab.strava_manifest import DEFAULT_DOWNLOAD_DIR, DEFAULT_MANIFEST, load_manifest, resolve_private_path, save_manifest
from endurance_lab.sync_lock import SyncAlreadyRunning, SyncLock
from endurance_lab.weekly_adjustment import adjust_week


@dataclass
class SyncOptions:
    date_from: date | None = None
    date_to: date | None = None
    all_history: bool = False
    dry_run: bool = False
    skip_coaching: bool = False
    skip_calendar: bool = False
    force_discovery: bool = False
    manifest: str | Path = DEFAULT_MANIFEST
    downloads: str | Path = DEFAULT_DOWNLOAD_DIR
    database: str | Path | None = None
    profile: str | Path | None = None
    delay_seconds: float = 2.0
    max_retries: int = 2
    backoff_seconds: float = 2.0
    timeout_seconds: float = 60.0
    trigger: str = "cli"


@dataclass
class SyncResult:
    status: str = "running"
    error_code: str | None = None
    error_summary: str | None = None
    window_from: str | None = None
    window_to: str | None = None
    discovery: dict[str, Any] = field(default_factory=dict)
    downloads: dict[str, Any] = field(default_factory=dict)
    imports: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    coaching: dict[str, Any] = field(default_factory=dict)
    calendar: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    run_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyncPipeline:
    def __init__(
        self,
        *,
        transport_factory: Callable[..., Any] = PlaywrightTransport,
        authentication_checker: Callable[..., Any] = check_authentication,
        discovery_factory: Callable[..., StravaDiscovery] = StravaDiscovery,
        downloader_factory: Callable[..., StravaDownloader] = StravaDownloader,
    ) -> None:
        self.transport_factory = transport_factory
        self.authentication_checker = authentication_checker
        self.discovery_factory = discovery_factory
        self.downloader_factory = downloader_factory

    def run(self, options: SyncOptions) -> SyncResult:
        result = SyncResult()
        logger = _logger()
        manifest = resolve_private_path(options.manifest, DEFAULT_MANIFEST)
        downloads = resolve_private_path(options.downloads, DEFAULT_DOWNLOAD_DIR)
        if options.all_history and options.date_from:
            raise ValueError("--all and --from cannot be used together")
        default_from, default_to = default_discovery_window(manifest, options.database)
        date_from = None if options.all_history else options.date_from or default_from
        date_to = options.date_to or default_to
        result.window_from = date_from.isoformat() if date_from else None
        result.window_to = date_to.isoformat()
        init_db(options.database)
        try:
            with SyncLock():
                resumable = None if options.dry_run or options.force_discovery else _resumable_coaching_run(
                    options.database, date_from, date_to
                )
                if resumable:
                    result = _resume_coaching(resumable, options)
                    logger.info("sync resumed coaching run_id=%s", result.run_id)
                    return result
                if not options.dry_run:
                    result.run_id = _new_run(options, date_from, date_to)
                logger.info("sync started run_id=%s from=%s to=%s", result.run_id, date_from, date_to)
                with self.transport_factory(options.profile, headless=True) as transport:
                    auth = self.authentication_checker(transport.context)
                    if not auth.authenticated:
                        raise DiscoveryError("AUTH_REQUIRED", auth.reason)
                    _stage(result.run_id, "authentication", "success", {"reason": auth.reason}, options.database)

                    discovery = self.discovery_factory(
                        transport,
                        delay_seconds=max(0.5, options.delay_seconds / 2),
                        max_retries=options.max_retries,
                        backoff_seconds=options.backoff_seconds,
                        timeout_seconds=options.timeout_seconds,
                    ).discover(
                        date_from=date_from,
                        date_to=date_to,
                        manifest_path=manifest,
                        database=options.database,
                        sync_run_id=result.run_id,
                        dry_run=options.dry_run,
                        force=options.force_discovery,
                    )
                    result.discovery = {
                        "found": discovery.found, "new": discovery.new,
                        "updated": discovery.updated, "existing": discovery.existing,
                        "pages_checked": discovery.pages_checked,
                        "total_pages": discovery.total_pages, "complete": discovery.complete,
                    }
                    _stage(result.run_id, "discovery", "success", result.discovery, options.database)

                    if options.dry_run:
                        result.downloads = {
                            "selected": discovery.found, "downloaded": 0,
                            "already_present": 0, "tcx_fallback": 0, "failed": 0,
                            "dry_run": True,
                        }
                        result.status = "dry_run"
                        return result

                    # Build the work set from the persisted manifest, not only the
                    # pages fetched in this process.  An interrupted discovery run
                    # may resume at page N, while pages 1..N-1 were already safely
                    # checkpointed to the manifest.
                    candidate_ids = _manifest_ids_in_window(manifest, date_from, date_to)
                    downloader = self.downloader_factory(
                        transport,
                        delay_seconds=options.delay_seconds,
                        max_retries=options.max_retries,
                        backoff_seconds=options.backoff_seconds,
                        timeout_seconds=options.timeout_seconds,
                        authentication_probe=lambda: self.authentication_checker(transport.context).authenticated,
                    )
                    download_summary = downloader.run(
                        manifest, downloads, dry_run=options.dry_run,
                        activity_ids=candidate_ids, emit=lambda line: logger.info("download %s", line),
                    )
                    result.downloads = {
                        "selected": download_summary.selected,
                        "downloaded": download_summary.downloaded,
                        "already_present": download_summary.already_present,
                        "tcx_fallback": download_summary.fallback_successes,
                        "failed": download_summary.failed,
                        "formats": dict(download_summary.formats),
                    }
                    if download_summary.authentication_required:
                        raise DiscoveryError("AUTH_REQUIRED", "Authentication expired during downloads")
                    if download_summary.rate_limited:
                        raise DiscoveryError("RATE_LIMITED", "Strava rate limit reached during downloads")
                    _stage(result.run_id, "downloads", "success", result.downloads, options.database)

                imported_ids, import_summary = _import_and_fallback(
                    manifest, downloads, candidate_ids, options.database
                )
                result.imports = import_summary
                result.downloads["unavailable_metadata_fallback"] = import_summary["metadata_only"]
                result.downloads["failed"] = max(0, int(result.downloads["failed"]) - import_summary["metadata_only"])
                _stage(result.run_id, "import", "success", result.imports, options.database)

                quality = ingestion_quality_report(options.database)
                result.quality = {
                    "total_activities": quality["total_activities"],
                    "suspicious_activities": len(quality["suspicious_activities"]),
                }
                _stage(result.run_id, "quality", "success", result.quality, options.database)

                if imported_ids:
                    analyzed = analyze_database(options.database, activity_ids=imported_ids)
                else:
                    analyzed = 0
                _stage(result.run_id, "analytics", "success", {"activities_recalculated": analyzed}, options.database)

                archive = reconcile(manifest, downloads, options.database)
                before_matches = _selected_matches(options.database)
                match_counts = match_training_plan(options.database)
                after_matches = _selected_matches(options.database)
                result.reconciliation = {
                    "newly_matched": max(0, after_matches - before_matches),
                    "selected_matches": after_matches,
                    "match_counts": match_counts,
                    "manifest": archive["manifest"],
                    "downloaded": archive["downloaded"],
                    "imported_matching_manifest": archive["imported_matching_manifest"],
                    "pending_downloads": len(archive["missing_download_ids"]),
                    "pending_imports": len(archive["downloaded_not_imported_ids"]),
                }
                _stage(result.run_id, "reconciliation", "success", result.reconciliation, options.database)

                try:
                    result.coaching = _update_coaching(imported_ids, options)
                    _stage(result.run_id, "coaching", "success", result.coaching, options.database)
                except Exception as exc:
                    _stage(result.run_id, "coaching", "failed", {}, options.database, "COACHING_ERROR", str(exc))
                    raise RuntimeError(f"coaching update failed: {exc}") from exc
                result.calendar = _update_calendar(result, options)
                result.status = "success"
                _finish_run(result, options.database)
                logger.info("sync success run_id=%s", result.run_id)
                return result
        except SyncAlreadyRunning as exc:
            result.status, result.error_code, result.error_summary = "failed", "SYNC_LOCKED", str(exc)
        except DiscoveryError as exc:
            result.status, result.error_code, result.error_summary = "stopped", exc.code, str(exc)
        except Exception as exc:
            result.status, result.error_code, result.error_summary = "failed", _classify_error(exc), str(exc)
        logger.error("sync stopped code=%s error=%s", result.error_code, result.error_summary)
        if result.run_id:
            _finish_run(result, options.database)
        return result


def sync_status(database: str | Path | None = None, manifest_path: str | Path = DEFAULT_MANIFEST,
                downloads_dir: str | Path = DEFAULT_DOWNLOAD_DIR) -> dict[str, Any]:
    init_db(database)
    manifest = resolve_private_path(manifest_path, DEFAULT_MANIFEST)
    directory = resolve_private_path(downloads_dir, DEFAULT_DOWNLOAD_DIR)
    rows = load_manifest(manifest)
    with connect(database) as connection:
        run = connection.execute("SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
        success = connection.execute(
            "SELECT * FROM sync_runs WHERE status='success' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest = connection.execute(
            "SELECT source_activity_id, started_at, name FROM activities ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        imported_source_ids = {
            str(row[0]) for row in connection.execute(
                "SELECT source_activity_id FROM activities WHERE source_activity_id IS NOT NULL"
            ).fetchall()
        }
        metadata_only = int(connection.execute("SELECT COUNT(1) FROM activities WHERE metadata_only=1").fetchone()[0])
        auth_stage = connection.execute(
            "SELECT status FROM sync_run_stages WHERE sync_run_id=? AND stage='authentication'",
            (run["id"],),
        ).fetchone() if run else None
    pending_downloads = sum(
        row.download_status not in {"downloaded", "metadata_only"}
        and find_valid_download(row, directory) is None for row in rows
    )
    pending_imports = sum(
        row.download_status == "downloaded"
        and row.import_status != "imported"
        and row.strava_id not in imported_source_ids
        for row in rows
    )
    return {
        "database": "OK",
        "last_run": dict(run) if run else None,
        "last_successful_sync": success["finished_at"] if success else None,
        "latest_activity": dict(latest) if latest else None,
        "pending_downloads": pending_downloads,
        "pending_imports": pending_imports,
        "metadata_only": metadata_only,
        "authentication": "OK" if auth_stage and auth_stage["status"] == "success" else (
            "REQUIRED" if run and run["error_code"] == "AUTH_REQUIRED" else "UNKNOWN"
        ),
    }


def _import_and_fallback(manifest, downloads, candidate_ids, database):
    rows = load_manifest(manifest)
    imported_ids: list[int] = []
    counts = {"new": 0, "upgraded": 0, "duplicates": 0, "metadata_only": 0, "errors": 0}
    for row in rows:
        if row.strava_id not in candidate_ids:
            continue
        _update_db_metadata(row, database)
        existing = find_valid_download(row, downloads)
        if existing:
            if row.import_status == "imported" and row.db_activity_id:
                continue
            outcome = import_file(existing[0], database)
            if outcome.status == "failed":
                counts["errors"] += 1
                row.import_status = "failed"
                row.error = outcome.message or "import failed"
                continue
            counts["new"] += outcome.imported_activities
            counts["upgraded"] += outcome.upgraded_activities
            counts["duplicates"] += outcome.skipped_activities
            activity_id = _db_activity_id(row.strava_id, database)
            if activity_id:
                imported_ids.append(activity_id)
                row.db_activity_id = str(activity_id)
                row.import_status = "imported"
                _update_db_metadata(row, database)
            continue
        if _export_unavailable(row) and row.activity_date:
            activity_id = upsert_metadata_activity(
                MetadataActivity(
                    row.strava_id, datetime.fromisoformat(row.activity_date), row.sport,
                    row.name or None, _number(row.duration_seconds), _number(row.distance_m),
                    source="strava_metadata", source_reference="strava_discovery",
                ),
                database,
            )
            row.download_status = "metadata_only"
            row.import_status = "imported"
            row.db_activity_id = str(activity_id)
            row.result_status = "original and TCX unavailable; metadata-only record stored"
            imported_ids.append(activity_id)
            counts["metadata_only"] += 1
            _update_db_metadata(row, database)
    save_manifest(rows, manifest)
    return sorted(set(imported_ids)), counts


def _export_unavailable(row) -> bool:
    text = f"{row.result_status} {row.error}".lower()
    return row.download_status == "failed" and (
        "http 404" in text or "not an activity export" in text or "unsupported binary" in text
    ) and "network error" not in text


def _new_run(options, date_from, date_to) -> int:
    with connect(options.database) as connection:
        cursor = connection.execute(
            """INSERT INTO sync_runs (started_at, trigger, status, requested_from, requested_to)
               VALUES (?, ?, 'running', ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), options.trigger,
             date_from.isoformat() if date_from else None, date_to.isoformat()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def _resumable_coaching_run(database, date_from, date_to):
    with connect(database) as connection:
        return connection.execute(
            """SELECT r.* FROM sync_runs r
               WHERE r.status='failed' AND r.error_code='COACHING_ERROR'
                 AND r.requested_from IS ? AND r.requested_to=?
                 AND EXISTS (SELECT 1 FROM sync_run_stages s WHERE s.sync_run_id=r.id AND s.stage='reconciliation' AND s.status='success')
                 AND NOT EXISTS (SELECT 1 FROM sync_run_stages s WHERE s.sync_run_id=r.id AND s.stage='coaching' AND s.status='success')
               ORDER BY r.id DESC LIMIT 1""",
            (date_from.isoformat() if date_from else None, date_to.isoformat()),
        ).fetchone()


def _manifest_ids_in_window(manifest, date_from, date_to) -> set[str]:
    selected: set[str] = set()
    for row in load_manifest(manifest):
        try:
            activity_day = date.fromisoformat(row.activity_date[:10])
        except (TypeError, ValueError):
            continue
        if activity_day > date_to or (date_from and activity_day < date_from):
            continue
        selected.add(row.strava_id)
    return selected


def _resume_coaching(run, options) -> SyncResult:
    try:
        prior = json.loads(run["summary_json"] or "{}")
    except json.JSONDecodeError:
        prior = {}
    result = SyncResult(
        status="running", run_id=int(run["id"]),
        window_from=run["requested_from"], window_to=run["requested_to"],
        discovery=prior.get("discovery") or {}, downloads=prior.get("downloads") or {},
        imports=prior.get("imports") or {}, quality=prior.get("quality") or {},
        reconciliation=prior.get("reconciliation") or {}, calendar=prior.get("calendar") or {},
        warnings=list(prior.get("warnings") or []),
    )
    with connect(options.database) as connection:
        connection.execute(
            "UPDATE sync_runs SET status='running', finished_at=NULL, error_code=NULL, error_summary=NULL WHERE id=?",
            (result.run_id,),
        )
        rows = connection.execute(
            "SELECT id FROM activities WHERE imported_at >= ? ORDER BY id", (run["started_at"],)
        ).fetchall()
        connection.commit()
    activity_ids = [int(row[0]) for row in rows]
    try:
        result.coaching = _update_coaching(activity_ids, options)
        _stage(result.run_id, "coaching", "success", result.coaching, options.database)
        result.calendar = _update_calendar(result, options)
        result.status = "success"
        _finish_run(result, options.database)
    except Exception as exc:
        result.status, result.error_code, result.error_summary = "failed", "COACHING_ERROR", str(exc)
        _stage(result.run_id, "coaching", "failed", {}, options.database, result.error_code, result.error_summary)
        _finish_run(result, options.database)
    return result


def _update_calendar(result: SyncResult, options: SyncOptions) -> dict[str, Any]:
    if options.skip_calendar or not calendar_enabled():
        value = {"enabled": False, "skipped": True}
        _stage(result.run_id, "calendar", "skipped", value, options.database)
        return value
    try:
        planned = sync_google_calendar(options.database).to_dict()
        settings = load_athlete_config().get("google_calendar", {})
        lookback = max(1, int(settings.get("actual_lookback_days", 30)))
        today = date.today()
        actual = sync_activity_calendar(
            options.database, date_from=today - timedelta(days=lookback), date_to=today
        ).to_dict()
        value = {
            "enabled": True,
            "skipped": False,
            "created": planned["created"] + actual["created"],
            "updated": planned["updated"] + actual["updated"],
            "unchanged": planned["unchanged"] + actual["unchanged"],
            "planned": planned,
            "activities": actual,
        }
        _stage(result.run_id, "calendar", "success", value, options.database)
        return value
    except GoogleCalendarError as exc:
        value = {"enabled": True, "error_code": exc.code, "error": str(exc)}
        _stage(result.run_id, "calendar", "failed", value, options.database, exc.code, str(exc))
        result.warnings.append(f"Google Calendar: {exc}")
        return value


def _update_coaching(activity_ids, options) -> dict[str, Any]:
    costs = [session_cost(activity_id, options.database, persist=True) for activity_id in activity_ids]
    prescriptions = []
    before_count = _prescription_count(options.database)
    if not options.skip_coaching:
        affected_weeks = {date.today()}
        affected_weeks.update(_activity_dates(activity_ids, options.database))
        for value in sorted(affected_weeks):
            prescriptions.extend(adjust_week(value, options.database, as_of=date.today(), persist=True))
    after_count = _prescription_count(options.database)
    return {
        "session_costs_updated": len(costs),
        "prescriptions_evaluated": len({item.prescription_id for item in prescriptions if item.prescription_id}),
        "prescriptions_updated": max(0, after_count - before_count),
        "skipped": options.skip_coaching,
    }


def _prescription_count(database) -> int:
    with connect(database) as connection:
        return int(connection.execute("SELECT COUNT(1) FROM workout_prescriptions").fetchone()[0])


def _stage(run_id, name, status, checkpoint, database, error_code=None, error_summary=None):
    if run_id is None:
        return
    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        connection.execute(
            """INSERT INTO sync_run_stages
               (sync_run_id, stage, status, started_at, finished_at, checkpoint_json, error_code, error_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(sync_run_id, stage) DO UPDATE SET status=excluded.status,
                   finished_at=excluded.finished_at, checkpoint_json=excluded.checkpoint_json,
                   error_code=excluded.error_code, error_summary=excluded.error_summary""",
            (run_id, name, status, now, now, json.dumps(checkpoint, default=str), error_code, error_summary),
        )
        connection.commit()


def _finish_run(result: SyncResult, database) -> None:
    with connect(database) as connection:
        connection.execute(
            """UPDATE sync_runs SET finished_at=?, status=?, discovered_count=?, downloaded_count=?,
                      metadata_only_count=?, imported_count=?, matched_count=?, prescriptions_updated=?,
                      warnings_count=?, error_code=?, error_summary=?, summary_json=? WHERE id=?""",
            (datetime.now(timezone.utc).isoformat(), result.status,
             int(result.discovery.get("new", 0)), int(result.downloads.get("downloaded", 0)),
             int(result.imports.get("metadata_only", 0)), int(result.imports.get("new", 0)) + int(result.imports.get("upgraded", 0)),
             int(result.reconciliation.get("newly_matched", 0)), int(result.coaching.get("prescriptions_updated", 0)),
             len(result.warnings), result.error_code, result.error_summary,
             json.dumps(result.to_dict(), default=str), result.run_id),
        )
        connection.commit()


def _selected_matches(database) -> int:
    with connect(database) as connection:
        return int(connection.execute("SELECT COUNT(1) FROM planned_activity_matches WHERE is_selected=1").fetchone()[0])


def _db_activity_id(strava_id, database) -> int | None:
    with connect(database) as connection:
        row = connection.execute("SELECT id FROM activities WHERE source_activity_id=?", (strava_id,)).fetchone()
        return int(row[0]) if row else None


def _update_db_metadata(row, database) -> None:
    with connect(database) as connection:
        existing = connection.execute(
            "SELECT id, metadata_only FROM activities WHERE source_activity_id=?", (row.strava_id,)
        ).fetchone()
        if not existing:
            return
        if int(existing["metadata_only"]):
            connection.execute(
                """UPDATE activities SET name=?, sport=?, started_at=?, elapsed_seconds=?,
                          moving_seconds=?, distance_m=? WHERE id=?""",
                (row.name or None, _normalized_sport(row.sport), row.activity_date,
                 _number(row.duration_seconds), _number(row.duration_seconds),
                 _number(row.distance_m), existing["id"]),
            )
        else:
            connection.execute(
                "UPDATE activities SET name=COALESCE(?, name), sport=? WHERE id=?",
                (row.name or None, _normalized_sport(row.sport), existing["id"]),
            )
        connection.commit()


def _activity_dates(activity_ids, database) -> list[date]:
    if not activity_ids:
        return []
    placeholders = ",".join("?" for _ in activity_ids)
    with connect(database) as connection:
        rows = connection.execute(
            f"SELECT DISTINCT substr(started_at,1,10) FROM activities WHERE id IN ({placeholders})",
            tuple(activity_ids),
        ).fetchall()
    return [date.fromisoformat(row[0]) for row in rows]


def _number(value) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _normalized_sport(value: str) -> str:
    from endurance_lab.plan_parser import normalize_plan_sport
    return normalize_plan_sport(value)


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "import" in text or "parse" in text:
        return "IMPORT_ERROR"
    if "coach" in text or "prescription" in text:
        return "COACHING_ERROR"
    return "SYNC_ERROR"


def _logger() -> logging.Logger:
    logger = logging.getLogger("endurance_lab.sync")
    if logger.handlers:
        return logger
    path = project_path("training_data/logs/sync.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
