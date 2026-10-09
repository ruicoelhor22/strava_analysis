"""Conservative reconciliation of the private Strava manifest and local archive."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from endurance_lab.analytics import analyze_database
from endurance_lab.db import connect, init_db
from endurance_lab.importer import import_file
from endurance_lab.metadata_activities import MetadataActivity, upsert_metadata_activity
from endurance_lab.plan_matching import match_training_plan
from endurance_lab.strava_export import find_valid_download
from endurance_lab.strava_manifest import (
    DEFAULT_DOWNLOAD_DIR, DEFAULT_MANIFEST, load_manifest, resolve_private_path, save_manifest,
)


def failure_reason(row) -> str:
    text = f"{row.result_status} {row.error}".lower()
    if row.download_status == "pending" and not text.strip():
        return "not_attempted"
    if row.download_status == "metadata_only":
        return "non_exportable"
    if "original: http 404" in text and "tcx: http 404" in text:
        return "non_exportable"
    if row.download_status == "authentication_required" or "authentication required" in text:
        return "authentication"
    if row.download_status == "rate_limited" or "http 429" in text:
        return "rate_limit"
    if "network error" in text or any(code in text for code in ("http 408", "http 500", "http 502", "http 503", "http 504")):
        return "network"
    if "html response is not an activity export" in text:
        return "ambiguous_html"
    if "invalid" in text or "unsupported binary" in text:
        return "invalid_export"
    if row.import_status == "failed":
        return "parse_failure"
    return "unknown"


def archive_status(
    database: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    downloads_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
) -> dict[str, Any]:
    init_db(database)
    rows = load_manifest(manifest_path)
    directory = resolve_private_path(downloads_dir, DEFAULT_DOWNLOAD_DIR)
    with connect(database) as connection:
        activities = {
            str(row["source_activity_id"]): dict(row)
            for row in connection.execute(
                """SELECT id,source_activity_id,metadata_only,started_at,name,sport,
                          elapsed_seconds,distance_m,device
                   FROM activities WHERE source_activity_id IS NOT NULL"""
            )
        }
        canonical = int(connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0])
        metadata_total = int(connection.execute("SELECT COUNT(*) FROM activities WHERE metadata_only=1").fetchone()[0])
    categories: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    missing_dates = 0
    seen: set[str] = set()
    duplicate_ids = 0
    non_strength_export_questions = 0
    for row in rows:
        if row.strava_id in seen:
            duplicate_ids += 1
            continue
        seen.add(row.strava_id)
        missing_dates += not bool(row.activity_date)
        activity = activities.get(row.strava_id)
        reason = failure_reason(row)
        if activity and not activity["metadata_only"]:
            categories["resolved_detailed"] += 1
        elif activity and activity["metadata_only"] and reason == "non_exportable":
            categories["resolved_metadata_only"] += 1
        elif activity and activity["metadata_only"]:
            categories["provisional_metadata_only"] += 1
            reasons[reason] += 1
            non_strength_export_questions += row.sport.lower() not in {"weighttraining", "strength"}
        elif find_valid_download(row, directory):
            categories["downloaded_not_imported"] += 1
            reasons["parse_failure" if row.import_status == "failed" else "import_pending"] += 1
        else:
            categories["unresolved"] += 1
            reasons[reason] += 1
    missing_activity_records = categories["downloaded_not_imported"] + categories["unresolved"]
    return {
        "manifest_entries": len(rows), "unique_ids": len(seen),
        "total_known_activities": canonical,
        "canonical_activities": canonical, "detailed_activities": canonical - metadata_total,
        "metadata_only_activities": metadata_total,
        "resolved_detailed": categories["resolved_detailed"],
        "resolved_metadata_only": categories["resolved_metadata_only"],
        "provisional_metadata_only": categories["provisional_metadata_only"],
        "downloaded_not_imported": categories["downloaded_not_imported"],
        "unresolved_without_metadata": categories["unresolved"],
        "missing_activity_records": missing_activity_records,
        "detail_recovery_candidates": categories["provisional_metadata_only"],
        "non_strength_export_questions": non_strength_export_questions,
        "missing_dates": missing_dates, "duplicate_manifest_ids": duplicate_ids,
        "failure_reasons": dict(sorted(reasons.items())),
        "detailed_data_coverage_percent": round(100 * (canonical - metadata_total) / canonical, 1) if canonical else 100.0,
        "status": "NEEDS_REVIEW" if missing_activity_records or duplicate_ids or missing_dates else "KNOWN",
    }


def reconcile_archive(
    database: str | Path | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    downloads_dir: str | Path = DEFAULT_DOWNLOAD_DIR,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repair locally provable links; do not infer non-exportability from HTML 200."""
    init_db(database)
    manifest = resolve_private_path(manifest_path, DEFAULT_MANIFEST)
    directory = resolve_private_path(downloads_dir, DEFAULT_DOWNLOAD_DIR)
    rows = load_manifest(manifest)
    before = archive_status(database, manifest, directory)
    actions: Counter[str] = Counter()
    with connect(database) as connection:
        inconsistent = connection.execute(
            """SELECT a.id FROM activities a
               WHERE a.metadata_only=1 AND a.source_format IN ('fit','tcx','json')
                 AND EXISTS (SELECT 1 FROM activity_sources s WHERE s.activity_id=a.id
                             AND s.selected=1 AND s.source_format IN ('fit','tcx','json'))"""
        ).fetchall()
        actions["repaired_detailed_flags"] = len(inconsistent)
        if not dry_run and inconsistent:
            connection.executemany(
                "UPDATE activities SET metadata_only=0 WHERE id=?",
                [(row["id"],) for row in inconsistent],
            )
            connection.commit()
    with connect(database) as connection:
        activities = {
            str(row["source_activity_id"]): dict(row)
            for row in connection.execute(
                """SELECT id,source_activity_id,metadata_only,started_at,name,sport,
                          elapsed_seconds,distance_m,device
                   FROM activities WHERE source_activity_id IS NOT NULL"""
            )
        }
    for row in rows:
        existing = activities.get(row.strava_id)
        if existing and _fill_from_activity(row, existing, dry_run=dry_run):
            actions["metadata_filled_from_local_db"] += 1
        valid = None if existing and not existing["metadata_only"] else find_valid_download(row, directory)
        if existing and not existing["metadata_only"]:
            if row.import_status != "imported" or row.db_activity_id != str(existing["id"]):
                actions["linked_existing_detailed"] += 1
                if not dry_run:
                    row.import_status = "imported"
                    row.db_activity_id = str(existing["id"])
            continue
        if valid:
            actions["valid_downloads_to_import"] += 1
            if not dry_run:
                outcome = import_file(valid[0], database)
                if outcome.status == "failed":
                    actions["import_failures"] += 1
                    row.import_status = "failed"
                else:
                    with connect(database) as connection:
                        match = connection.execute(
                            "SELECT id FROM activities WHERE source_activity_id=?", (row.strava_id,)
                        ).fetchone()
                    if match:
                        row.import_status = "imported"
                        row.db_activity_id = str(match["id"])
                        row.download_status = "downloaded"
                        row.filename = valid[0].name
            continue
        if not row.activity_date:
            continue
        try:
            started = datetime.fromisoformat(row.activity_date.replace("Z", "+00:00"))
        except ValueError:
            actions["invalid_dates"] += 1
            continue
        # Discovery establishes a real activity even when the old export result
        # is ambiguous. Store a real activity summary; its export detail remains unconfirmed.
        reason = failure_reason(row)
        if not existing:
            actions["metadata_candidates"] += 1
            if not dry_run:
                activity_id = upsert_metadata_activity(MetadataActivity(
                    row.strava_id, started, row.sport or "other", row.name or None,
                    _number(row.duration_seconds), _number(row.distance_m),
                    source="strava_metadata", source_reference="strava_discovery",
                ), database)
                with connect(database) as connection:
                    linked = connection.execute(
                        "SELECT metadata_only FROM activities WHERE id=?", (activity_id,)
                    ).fetchone()
                actions["linked_cross_source"] += int(linked is not None and not linked["metadata_only"])
                actions["metadata_created"] += int(linked is not None and bool(linked["metadata_only"]))
                row.db_activity_id = str(activity_id)
                row.import_status = "imported"
        elif row.import_status != "imported" or row.db_activity_id != str(existing["id"]):
            actions["linked_existing_metadata"] += 1
            if not dry_run:
                row.import_status = "imported"
                row.db_activity_id = str(existing["id"])
        if reason == "non_exportable":
            actions["confirmed_non_exportable"] += 1
            if not dry_run:
                row.download_status = "metadata_only"
    if not dry_run:
        save_manifest(rows, manifest)
    with connect(database) as connection:
        pending_analytics = [int(row[0]) for row in connection.execute(
            """SELECT a.id FROM activities a LEFT JOIN derived_activity_metrics d ON d.activity_id=a.id
               WHERE d.activity_id IS NULL OR (a.metadata_only=0 AND d.computed_at<a.imported_at)
               ORDER BY a.id"""
        )]
    actions["activities_missing_analytics"] = len(pending_analytics)
    if not dry_run and pending_analytics:
        actions["activities_analyzed"] = analyze_database(database, activity_ids=pending_analytics)
        match_training_plan(database)
    return {"before": before, "actions": dict(actions),
            "after": before if dry_run else archive_status(database, manifest, directory),
            "dry_run": dry_run}


def _number(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _fill_from_activity(row, activity: dict[str, Any], *, dry_run: bool) -> bool:
    fields = {
        "activity_date": activity.get("started_at"),
        "name": activity.get("name"),
        "sport": activity.get("sport"),
        "duration_seconds": activity.get("elapsed_seconds"),
        "distance_m": activity.get("distance_m"),
        "source_device": activity.get("device"),
    }
    missing = {key: value for key, value in fields.items() if not getattr(row, key) and value is not None}
    if not dry_run:
        for key, value in missing.items():
            setattr(row, key, str(value))
    return bool(missing)


def database_integrity(database: str | Path | None = None) -> dict[str, Any]:
    init_db(database)
    with connect(database) as connection:
        sqlite_result = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        checks = {
            "orphan_activity_sources": "SELECT COUNT(*) FROM activity_sources s LEFT JOIN activities a ON a.id=s.activity_id WHERE a.id IS NULL",
            "orphan_strength_sets": "SELECT COUNT(*) FROM strength_sets s LEFT JOIN activities a ON a.id=s.activity_id WHERE a.id IS NULL",
            "orphan_plan_matches": "SELECT COUNT(*) FROM planned_activity_matches m LEFT JOIN activities a ON a.id=m.activity_id WHERE m.activity_id IS NOT NULL AND a.id IS NULL",
            "orphan_prescriptions": "SELECT COUNT(*) FROM workout_prescriptions p LEFT JOIN planned_sessions s ON s.id=p.planned_session_id WHERE s.id IS NULL",
            "duplicate_source_ids": "SELECT COUNT(*) FROM (SELECT source_activity_id FROM activities WHERE source_activity_id IS NOT NULL GROUP BY source_activity_id HAVING COUNT(*)>1)",
            "duplicate_canonical_candidates": "SELECT COUNT(*) FROM (SELECT started_at,sport,COUNT(*) n FROM activities GROUP BY started_at,sport HAVING n>1)",
            "missing_source_provenance": "SELECT COUNT(*) FROM activities a WHERE NOT EXISTS (SELECT 1 FROM activity_sources s WHERE s.activity_id=a.id)",
            "metadata_flag_conflicts": """SELECT COUNT(*) FROM activities a WHERE a.metadata_only=1 AND EXISTS
                (SELECT 1 FROM activity_sources s WHERE s.activity_id=a.id AND s.selected=1
                 AND s.source_format IN ('fit','tcx','json'))""",
        }
        counts = {name: int(connection.execute(query).fetchone()[0]) for name, query in checks.items()}
        dates = [str(row[0]) for row in connection.execute("SELECT started_at FROM activities")]
    invalid_dates = future_dates = 0
    latest_allowed = datetime.now(timezone.utc) + timedelta(days=1)
    for value in dates:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            future_dates += parsed > latest_allowed
        except ValueError:
            invalid_dates += 1
    counts.update({"invalid_dates": invalid_dates, "future_dates": future_dates})
    return {
        "sqlite_integrity": sqlite_result, "foreign_key_violations": foreign_keys,
        **counts,
        "ok": sqlite_result == "ok" and foreign_keys == 0 and not any(counts.values()),
    }
