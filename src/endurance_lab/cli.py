from __future__ import annotations

import argparse
import getpass
import json
import sys
from dataclasses import asdict
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from endurance_lab.analytics import analyze_database
from endurance_lab.automation import (
    automation_status, configure_ngrok_phone_access, install_windows_tasks,
    phone_access_status, uninstall_windows_tasks,
)
from endurance_lab.athlete_state import athlete_state
from endurance_lab.coaching_backtest import backtest
from endurance_lab.config import ensure_local_layout, paths
from endurance_lab.db import init_db
from endurance_lab.google_calendar import (
    DEFAULT_CLIENT_FILE, DEFAULT_TOKEN_FILE, GoogleCalendarError, calendar_status,
    configure_google_calendars, login_google_calendar, sync_activity_calendar,
    sync_google_calendar,
)
from endurance_lab.importer import SUPPORTED_FORMATS, discover_files, import_path
from endurance_lab.quality import ingestion_quality_report
from endurance_lab.queries import data_quality
from endurance_lab.plan_adherence import plan_status, reconcile_plan
from endurance_lab.plan_importer import format_import_summary, import_training_plan
from endurance_lab.plan_matching import match_training_plan
from endurance_lab.plan_parser import inspect_workbook
from endurance_lab.prescription import prescribe, prescribe_week
from endurance_lab.session_classifier import classify_activity
from endurance_lab.session_cost import session_cost
from endurance_lab.weekly_adjustment import explain_date, reconcile_coaching
from endurance_lab.workout_execution import evaluate_activity
from endurance_lab.strava_auth import (
    AUTH_REQUIRED_MESSAGE,
    PlaywrightTransport,
    check_authentication,
    interactive_login,
    profile_path,
    reset_authentication_profile,
)
from endurance_lab.strava_export import (
    StravaDownloader,
    build_authenticated_session,
    format_download_summary,
    format_reconciliation,
    reconcile,
)
from endurance_lab.strava_manifest import merge_manifest_source
from endurance_lab.strava_discovery import DiscoveryError, StravaDiscovery, default_discovery_window
from endurance_lab.sync_pipeline import SyncOptions, SyncPipeline, sync_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="endurance-lab", description="Private multi-format endurance analytics"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create local directories, config, and database")
    import_command = commands.add_parser("import", help="Import TCX, FIT, JSON, or a mixed directory")
    import_command.add_argument("source", nargs="?", help="Defaults to data/import")
    commands.add_parser("analyze", help="Recalculate all derived metrics")
    rebuild = commands.add_parser("rebuild", help="Import supported files and recalculate metrics")
    rebuild.add_argument("source", nargs="?", help="Defaults to data/import")
    commands.add_parser("status", help="Show local database counts")
    commands.add_parser("quality", help="Report ingestion coverage and suspicious activities")
    plan_inspect = commands.add_parser(
        "plan-inspect", help="Inspect the private coaching workbook structure"
    )
    plan_inspect.add_argument("workbook", nargs="?", help="Auto-detected under training_data when omitted")
    plan_inspect.add_argument("--json", action="store_true")
    plan_import = commands.add_parser(
        "plan-import", help="Import the private coaching workbook into the normalized plan model"
    )
    plan_import.add_argument("workbook", nargs="?", help="Auto-detected under training_data when omitted")
    plan_import.add_argument("--database")
    plan_match = commands.add_parser(
        "plan-match", help="Match planned sessions to imported activities"
    )
    plan_match.add_argument("--database")
    plan_status_command = commands.add_parser(
        "plan-status", help="Summarize plan coverage, matches, and the current week"
    )
    plan_status_command.add_argument("--database")
    plan_status_command.add_argument("--json", action="store_true")
    plan_reconcile = commands.add_parser(
        "plan-reconcile", help="Report unresolved plan/match/data-confidence discrepancies"
    )
    plan_reconcile.add_argument("--database")
    plan_reconcile.add_argument("--json", action="store_true")
    state_command = commands.add_parser(
        "athlete-state", help="Build date-bounded athlete training context"
    )
    state_command.add_argument("date", nargs="?", help="YYYY-MM-DD; defaults to today")
    state_command.add_argument("--database")
    state_command.add_argument("--json", action="store_true")
    coach_today = commands.add_parser(
        "coach-today", help="Generate and persist today's plan-anchored prescription"
    )
    coach_today.add_argument("--database")
    coach_today.add_argument("--json", action="store_true")
    coach_date = commands.add_parser(
        "coach-date", help="Generate and persist a prescription for YYYY-MM-DD"
    )
    coach_date.add_argument("date")
    coach_date.add_argument("--database")
    coach_date.add_argument("--json", action="store_true")
    coach_week = commands.add_parser(
        "coach-week", help="Generate plan-anchored prescriptions for a calendar week"
    )
    coach_week.add_argument("date", nargs="?", help="Any date in the week; defaults to today")
    coach_week.add_argument("--database")
    coach_week.add_argument("--json", action="store_true")
    evaluation = commands.add_parser(
        "evaluate-activity", help="Evaluate execution of a matched completed activity"
    )
    evaluation.add_argument("activity_id", type=int)
    evaluation.add_argument("--database")
    evaluation.add_argument("--json", action="store_true")
    backtest_command = commands.add_parser(
        "coach-backtest", help="Backtest coaching rules without future activity data"
    )
    backtest_command.add_argument("--from", dest="date_from", required=True)
    backtest_command.add_argument("--to", dest="date_to", required=True)
    backtest_command.add_argument("--database")
    backtest_command.add_argument("--json", action="store_true")
    sync_command = commands.add_parser("sync", help="Bring Strava, analytics, plan matching, and coaching up to date")
    sync_command.add_argument("--from", dest="date_from")
    sync_command.add_argument("--to", dest="date_to")
    sync_command.add_argument("--all", action="store_true", dest="all_history")
    sync_command.add_argument("--dry-run", action="store_true")
    sync_command.add_argument("--skip-coaching", action="store_true")
    sync_command.add_argument("--skip-calendar", action="store_true")
    sync_command.add_argument("--force-discovery", action="store_true")
    sync_command.add_argument("--profile", default="training_data/auth/strava-browser-profile")
    sync_command.add_argument("--manifest", default="training_data/manifest.csv")
    sync_command.add_argument("--output", default="training_data/import")
    sync_command.add_argument("--database")
    sync_command.add_argument("--delay", type=float, default=2.0)
    sync_command.add_argument("--json", action="store_true")
    discover_command = commands.add_parser("strava-discover", help="Discover Strava activities without manual ID files")
    discover_command.add_argument("--from", dest="date_from")
    discover_command.add_argument("--to", dest="date_to")
    discover_command.add_argument("--all", action="store_true", dest="all_history")
    discover_command.add_argument("--dry-run", action="store_true")
    discover_command.add_argument("--force", action="store_true")
    discover_command.add_argument("--profile", default="training_data/auth/strava-browser-profile")
    discover_command.add_argument("--manifest", default="training_data/manifest.csv")
    discover_command.add_argument("--database")
    discover_command.add_argument("--json", action="store_true")
    sync_status_command = commands.add_parser("sync-status", help="Show sync, queue, and local freshness status")
    sync_status_command.add_argument("--manifest", default="training_data/manifest.csv")
    sync_status_command.add_argument("--downloads", default="training_data/import")
    sync_status_command.add_argument("--database")
    sync_status_command.add_argument("--json", action="store_true")
    calendar_login = commands.add_parser("calendar-login", help="Authorize private Google Calendar access")
    calendar_login.add_argument("--client", default=DEFAULT_CLIENT_FILE)
    calendar_login.add_argument("--token", default=DEFAULT_TOKEN_FILE)
    calendar_setup = commands.add_parser("calendar-setup", help="Assign separate actual and planned calendars")
    calendar_setup.add_argument("--actual", default="strava", help="Existing completed-activity calendar name")
    calendar_setup.add_argument("--planned", default="Endurance Lab - Planned", help="Planned-session calendar name")
    calendar_sync = commands.add_parser("calendar-sync", help="Create or update upcoming prescribed workouts")
    calendar_sync.add_argument("--days", type=int)
    calendar_sync.add_argument("--database")
    calendar_sync.add_argument("--dry-run", action="store_true")
    calendar_sync.add_argument("--json", action="store_true")
    calendar_status_command = commands.add_parser("calendar-status", help="Show Google Calendar connection state")
    calendar_status_command.add_argument("--database")
    calendar_status_command.add_argument("--json", action="store_true")
    activity_calendar = commands.add_parser(
        "calendar-activities", help="Create or update completed activities in their Google calendar"
    )
    activity_calendar.add_argument("--from", dest="date_from", required=True, type=_parse_date)
    activity_calendar.add_argument("--to", dest="date_to", required=True, type=_parse_date)
    activity_calendar.add_argument("--database")
    activity_calendar.add_argument("--dry-run", action="store_true")
    activity_calendar.add_argument("--json", action="store_true")
    automation_install = commands.add_parser("automation-install", help="Install Windows sync and dashboard tasks")
    automation_install.add_argument("--sync-minutes", type=int, default=60)
    automation_install.add_argument("--dry-run", action="store_true")
    automation_remove = commands.add_parser("automation-remove", help="Remove Windows automation tasks")
    automation_remove.add_argument("--dry-run", action="store_true")
    automation_status_command = commands.add_parser("automation-status", help="Show tasks and private phone access")
    automation_status_command.add_argument("--json", action="store_true")
    phone_status_command = commands.add_parser("phone-access", help="Show the private phone dashboard address")
    phone_status_command.add_argument("--json", action="store_true")
    phone_setup = commands.add_parser("phone-access-setup", help="Configure portable authenticated ngrok access")
    phone_setup.add_argument("--token-file", help="Private text file containing the ngrok authtoken")
    phone_setup.add_argument("--username", default="endurance")
    classification = commands.add_parser(
        "classify-activity", help="Classify the multidimensional cost of an imported activity"
    )
    classification.add_argument("activity_id", type=int)
    classification.add_argument("--database")
    classification.add_argument("--json", action="store_true")
    cost_command = commands.add_parser(
        "session-cost", help="Show the persisted cost dimensions and evidence for an activity"
    )
    cost_command.add_argument("activity_id", type=int)
    cost_command.add_argument("--database")
    cost_command.add_argument("--json", action="store_true")
    coach_reconcile = commands.add_parser(
        "coach-reconcile", help="Reconcile a calendar week with completed training"
    )
    coach_reconcile.add_argument("date", nargs="?", help="Any date in the week; defaults to today")
    coach_reconcile.add_argument("--database")
    coach_reconcile.add_argument("--json", action="store_true")
    coach_explain = commands.add_parser(
        "coach-explain", help="Explain the coaching decision for one date"
    )
    coach_explain.add_argument("date")
    coach_explain.add_argument("--database")
    coach_explain.add_argument("--json", action="store_true")
    manifest_command = commands.add_parser(
        "strava-manifest", help="Merge an ID text file or metadata CSV into the private manifest"
    )
    manifest_command.add_argument("source", help="Text/CSV file containing Strava activity IDs")
    manifest_command.add_argument("--manifest", default="training_data/manifest.csv")
    login = commands.add_parser(
        "strava-login", help="Open visible Chromium for manual Strava authentication"
    )
    login.add_argument("--profile", default="training_data/auth/strava-browser-profile")
    auth_status = commands.add_parser(
        "strava-auth-status", help="Check the persistent local Strava browser session"
    )
    auth_status.add_argument("--profile", default="training_data/auth/strava-browser-profile")
    auth_status.add_argument("--headed", action="store_true", help="Show Chromium while checking")
    auth_reset = commands.add_parser(
        "strava-auth-reset", help="Remove the private browser profile after typed confirmation"
    )
    auth_reset.add_argument("--profile", default="training_data/auth/strava-browser-profile")
    download = commands.add_parser(
        "strava-download", help="Download validated activity exports from a private manifest"
    )
    download.add_argument("--manifest", default="training_data/manifest.csv")
    download.add_argument("--output", default="training_data/import")
    download.add_argument("--auth", choices=("playwright", "cookies"), default="playwright")
    download.add_argument("--profile", default="training_data/auth/strava-browser-profile")
    download.add_argument("--headed", action="store_true", help="Show Chromium during downloading")
    download.add_argument(
        "--cookies", default="training_data/auth/cookies.txt",
        help="Netscape cookies file used only with --auth cookies",
    )
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--limit", type=int, help="Process at most this many missing activities")
    download.add_argument("--delay", type=float, default=2.0, help="Seconds between requests")
    download.add_argument("--retries", type=int, default=2, help="Retries for transient failures")
    download.add_argument("--backoff", type=float, default=2.0, help="Initial retry backoff seconds")
    download.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout seconds")
    download.add_argument("--force", action="store_true", help="Recheck the server even when a valid file exists")
    reconciliation = commands.add_parser(
        "strava-reconcile", help="Compare the private manifest, downloads, and imported database"
    )
    reconciliation.add_argument("--manifest", default="training_data/manifest.csv")
    reconciliation.add_argument("--downloads", default="training_data/import")
    reconciliation.add_argument("--database")
    reconciliation.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        layout = ensure_local_layout()
        init_db(layout.database)
        print(f"Database ready: {layout.database}")
        print(f"Activity import directory: {layout.import_dir}")
        return 0
    if args.command in {"import", "rebuild"}:
        ensure_local_layout()
        files = discover_files(getattr(args, "source", None))
        print(f"Scanning {len(files)} files...")
        results = import_path(getattr(args, "source", None))
        if not results:
            print(f"No files found in {getattr(args, 'source', None) or paths().import_dir}")
        for result in results:
            detail = f" ({result.message})" if result.message else ""
            print(
                f"{result.status:11} [{result.source_format.upper():4}] {result.file.name}: "
                f"{result.imported_activities} imported, {result.upgraded_activities} upgraded, "
                f"{result.skipped_activities} skipped{detail}"
            )
        formats = Counter(result.source_format for result in results if f".{result.source_format}" in SUPPORTED_FORMATS)
        statuses = Counter(result.status for result in results)
        print("Formats: " + ", ".join(f"{key.upper()} {value}" for key, value in sorted(formats.items())))
        print(
            "Activities: "
            f"imported {sum(result.imported_activities for result in results)}, "
            f"upgraded {sum(result.upgraded_activities for result in results)}, "
            f"skipped/duplicates {sum(result.skipped_activities for result in results)}"
        )
        print("Files: " + ", ".join(f"{key} {value}" for key, value in sorted(statuses.items())))
        had_failures = bool(statuses.get("failed"))
        if args.command == "import":
            return 1 if had_failures else 0
    if args.command in {"analyze", "rebuild"}:
        count = analyze_database()
        print(f"Recalculated analytics for {count} activities")
        return 1 if locals().get("had_failures", False) else 0
    if args.command == "status":
        init_db()
        quality = data_quality()
        for key, value in quality.items():
            print(f"{key}: {value}")
        return 0
    if args.command == "quality":
        print(json.dumps(ingestion_quality_report(), indent=2, default=str))
        return 0
    if args.command == "plan-inspect":
        try:
            report = inspect_workbook(args.workbook)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Plan inspection error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print("TRAINING PLAN WORKBOOK")
            print(f"Workbook: {report['workbook']}")
            print(f"Calendar header row: {report['calendar_header_row']}")
            for sheet in report["sheets"]:
                print(
                    f"- {sheet['name']}: {sheet['dimensions']}, "
                    f"{sheet['nonempty_cells']} populated cells, {sheet['formulas']} formulas, "
                    f"{len(sheet['merged_ranges'])} merged ranges"
                )
        return 0
    if args.command == "plan-import":
        try:
            summary = import_training_plan(args.workbook, args.database)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Plan import error: {exc}", file=sys.stderr)
            return 2
        print(format_import_summary(summary))
        return 0
    if args.command == "plan-match":
        counts = match_training_plan(args.database)
        print("PLAN MATCHING")
        for key in ("matched", "probable", "ambiguous", "unmatched", "manual"):
            print(f"{key.title()}: {counts.get(key, 0)}")
        return 0
    if args.command == "plan-status":
        report = plan_status(args.database)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            plan = report.get("plan") or {}
            print("TRAINING PLAN")
            print(f"Date range: {plan.get('date_start', '—')} to {plan.get('date_end', '—')}")
            print(f"Planned sessions: {report['sessions']}")
            for key in ("matched", "manual", "probable", "ambiguous", "unmatched"):
                print(f"{key.title()}: {report['matches'].get(key, 0)}")
            print(f"Completed: {report['states'].get('completed', 0) + report['states'].get('completed_metadata', 0)}")
            print(f"Upcoming: {report['states'].get('upcoming', 0)}")
            print(f"Missed: {report['states'].get('missed', 0)}")
            print("\nCURRENT WEEK")
            week = report.get("current_week")
            if not week:
                print("No planned sessions this week.")
            else:
                print(f"Planned: {week['planned_sessions']}")
                print(f"Completed: {week['completed_sessions']}")
                print(
                    "Duration: "
                    f"{week['actual_duration_seconds'] / 3600:.1f} h actual / "
                    f"{week['planned_duration_seconds'] / 3600:.1f} h planned"
                )
                print(
                    "Load: "
                    f"{week['actual_load']:.0f} actual / "
                    f"{week['planned_load']:.0f} planned"
                    if week["planned_load"] is not None else
                    f"Load: {week['actual_load']:.0f} actual / unavailable planned"
                )
        return 0
    if args.command == "plan-reconcile":
        report = reconcile_plan(args.database)
        print(json.dumps(report, indent=2, default=str) if args.json else _format_plan_reconciliation(report))
        return 1 if report["ambiguous"] or report["unmatched"] else 0
    if args.command == "athlete-state":
        try:
            value = _parse_date(args.date) if args.date else date.today()
            state = athlete_state(value, args.database)
        except ValueError as exc:
            print(f"Athlete-state error: {exc}", file=sys.stderr)
            return 2
        payload = state.to_dict()
        print(json.dumps(payload, indent=2, default=str) if args.json else _format_athlete_state(payload))
        return 0
    if args.command in {"coach-today", "coach-date"}:
        try:
            value = date.today() if args.command == "coach-today" else _parse_date(args.date)
            result = prescribe(value, args.database, persist=True)
        except ValueError as exc:
            print(f"Coaching error: {exc}", file=sys.stderr)
            return 2
        payload = result.to_dict()
        print(json.dumps(payload, indent=2, default=str) if args.json else _format_prescription(payload))
        return 0
    if args.command == "coach-week":
        try:
            value = _parse_date(args.date) if args.date else date.today()
            results = [item.to_dict() for item in prescribe_week(value, args.database, persist=True)]
        except ValueError as exc:
            print(f"Coaching error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(results, indent=2, default=str) if args.json else _format_week(results, value))
        return 0
    if args.command == "evaluate-activity":
        try:
            result = evaluate_activity(args.activity_id, args.database, persist=True).to_dict()
        except ValueError as exc:
            print(f"Activity evaluation error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str) if args.json else _format_evaluation(result))
        return 0
    if args.command == "coach-backtest":
        try:
            result = backtest(_parse_date(args.date_from), _parse_date(args.date_to), args.database)
        except ValueError as exc:
            print(f"Backtest error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str) if args.json else _format_backtest(result))
        return 0
    if args.command == "sync":
        try:
            result = SyncPipeline().run(SyncOptions(
                date_from=_parse_date(args.date_from) if args.date_from else None,
                date_to=_parse_date(args.date_to) if args.date_to else None,
                all_history=args.all_history, dry_run=args.dry_run,
                skip_coaching=args.skip_coaching, skip_calendar=args.skip_calendar,
                force_discovery=args.force_discovery,
                profile=args.profile, manifest=args.manifest, downloads=args.output,
                database=args.database, delay_seconds=args.delay,
            ))
        except ValueError as exc:
            print(f"Sync error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result.to_dict(), indent=2, default=str) if args.json else _format_sync(result.to_dict()))
        if result.status in {"success", "dry_run"}:
            return 0
        return 3 if result.error_code == "AUTH_REQUIRED" else 4 if result.error_code == "SYNC_LOCKED" else 1
    if args.command == "strava-discover":
        try:
            if args.all_history and args.date_from:
                raise ValueError("--all and --from cannot be used together")
            default_from, default_to = default_discovery_window(args.manifest, args.database)
            start = None if args.all_history else (_parse_date(args.date_from) if args.date_from else default_from)
            end = _parse_date(args.date_to) if args.date_to else default_to
            with PlaywrightTransport(args.profile, headless=True) as transport:
                authentication = check_authentication(transport.context)
                if not authentication.authenticated:
                    print(AUTH_REQUIRED_MESSAGE, file=sys.stderr)
                    return 3
                result = StravaDiscovery(transport).discover(
                    date_from=start, date_to=end, manifest_path=args.manifest,
                    database=args.database, dry_run=args.dry_run, force=args.force,
                )
        except (ValueError, RuntimeError, DiscoveryError) as exc:
            print(f"Discovery error: {exc}", file=sys.stderr)
            return 3 if isinstance(exc, DiscoveryError) and exc.code == "AUTH_REQUIRED" else 1
        payload = {
            "from": start.isoformat() if start else None, "to": end.isoformat(),
            "found": result.found, "new": result.new, "updated": result.updated,
            "existing": result.existing, "pages_checked": result.pages_checked,
            "total_pages": result.total_pages, "complete": result.complete,
        }
        print(json.dumps(payload, indent=2) if args.json else _format_discovery(payload))
        return 0
    if args.command == "sync-status":
        report = sync_status(args.database, args.manifest, args.downloads)
        print(json.dumps(report, indent=2, default=str) if args.json else _format_sync_status(report))
        return 0
    if args.command == "calendar-login":
        try:
            report = login_google_calendar(args.client, args.token)
        except GoogleCalendarError as exc:
            print(f"Google Calendar authorization error: {exc}", file=sys.stderr)
            return 5
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "calendar-setup":
        try:
            report = configure_google_calendars(args.actual, args.planned)
        except GoogleCalendarError as exc:
            print(f"Google Calendar setup error: {exc}", file=sys.stderr)
            return 5
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "calendar-sync":
        try:
            report = sync_google_calendar(
                args.database, days_ahead=args.days, dry_run=args.dry_run
            ).to_dict()
        except GoogleCalendarError as exc:
            print(f"Google Calendar sync error: {exc}", file=sys.stderr)
            return 5
        print(json.dumps(report, indent=2) if args.json else _format_calendar_sync(report))
        return 1 if report["errors"] else 0
    if args.command == "calendar-status":
        report = calendar_status(args.database)
        print(json.dumps(report, indent=2) if args.json else _format_calendar_status(report))
        return 0
    if args.command == "calendar-activities":
        try:
            report = sync_activity_calendar(
                args.database, date_from=args.date_from, date_to=args.date_to,
                dry_run=args.dry_run,
            ).to_dict()
        except (GoogleCalendarError, ValueError) as exc:
            print(f"Google Calendar activity sync error: {exc}", file=sys.stderr)
            return 5
        print(json.dumps(report, indent=2) if args.json else _format_activity_calendar_sync(report))
        return 1 if report["errors"] else 0
    if args.command == "automation-install":
        try:
            report = asdict(install_windows_tasks(args.sync_minutes, dry_run=args.dry_run))
        except RuntimeError as exc:
            print(f"Automation installation error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "automation-remove":
        report = uninstall_windows_tasks(dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "automation-status":
        report = automation_status()
        print(json.dumps(report, indent=2) if args.json else _format_automation_status(report))
        return 0
    if args.command == "phone-access":
        report = phone_access_status()
        print(json.dumps(report, indent=2) if args.json else _format_phone_access(report))
        return 0
    if args.command == "phone-access-setup":
        try:
            token = (
                Path(args.token_file).read_text(encoding="utf-8").strip()
                if args.token_file else getpass.getpass("ngrok authtoken (input hidden): ").strip()
            )
            report = configure_ngrok_phone_access(token, args.username)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Phone access setup error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "classify-activity":
        try:
            result = classify_activity(args.activity_id, args.database).to_dict()
        except ValueError as exc:
            print(f"Classification error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str) if args.json else _format_classification(result))
        return 0
    if args.command == "session-cost":
        try:
            result = session_cost(args.activity_id, args.database, persist=True).to_dict()
        except ValueError as exc:
            print(f"Session-cost error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str) if args.json else _format_session_cost(result))
        return 0
    if args.command == "coach-reconcile":
        try:
            value = _parse_date(args.date) if args.date else date.today()
            result = reconcile_coaching(args.database, as_of=value, persist=True)
        except ValueError as exc:
            print(f"Coaching reconciliation error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str) if args.json else _format_coaching_reconciliation(result))
        return 0
    if args.command == "coach-explain":
        try:
            result = explain_date(_parse_date(args.date), args.database)
        except ValueError as exc:
            print(f"Coaching explanation error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str) if args.json else _format_coaching_explanation(result))
        return 0
    if args.command == "strava-manifest":
        try:
            target, added, updated = merge_manifest_source(args.source, args.manifest)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Manifest error: {exc}", file=sys.stderr)
            return 2
        print(f"Manifest: {target}")
        print(f"Added: {added}; metadata updated: {updated}")
        return 0
    if args.command == "strava-login":
        try:
            status = interactive_login(args.profile)
        except (RuntimeError, ValueError) as exc:
            print(f"Strava login error: {exc}", file=sys.stderr)
            return 2
        return 0 if status.authenticated else 1
    if args.command == "strava-auth-status":
        try:
            resolved = profile_path(args.profile)
        except ValueError as exc:
            print(f"Authentication check error: {exc}", file=sys.stderr)
            return 2
        print("Strava authentication")
        print(f"Profile: {resolved}")
        print(f"Profile exists: {'yes' if resolved.is_dir() else 'no'}")
        if not resolved.is_dir():
            print("Authenticated: no")
            print(AUTH_REQUIRED_MESSAGE)
            return 1
        try:
            with PlaywrightTransport(resolved, headless=not args.headed) as transport:
                status = check_authentication(transport.context)
        except RuntimeError as exc:
            print(f"Authentication check error: {exc}", file=sys.stderr)
            return 2
        print(f"Authenticated: {'yes' if status.authenticated else 'no'}")
        print(f"Check: {status.reason}")
        if not status.authenticated:
            print(AUTH_REQUIRED_MESSAGE)
        return 0 if status.authenticated else 1
    if args.command == "strava-auth-reset":
        try:
            removed = reset_authentication_profile(args.profile)
        except ValueError as exc:
            print(f"Authentication reset error: {exc}", file=sys.stderr)
            return 2
        return 0 if removed else 1
    if args.command == "strava-download":
        if args.limit is not None and args.limit < 1:
            print("Download error: --limit must be positive", file=sys.stderr)
            return 2
        try:
            if args.dry_run:
                summary = _run_strava_downloader(None, args)
            elif args.auth == "cookies":
                summary = _run_strava_downloader(build_authenticated_session(args.cookies), args)
            else:
                resolved = profile_path(args.profile)
                if not resolved.is_dir():
                    print(AUTH_REQUIRED_MESSAGE, file=sys.stderr)
                    return 2
                with PlaywrightTransport(resolved, headless=not args.headed) as transport:
                    authentication = check_authentication(transport.context)
                    if not authentication.authenticated:
                        print(f"Authentication check: {authentication.reason}", file=sys.stderr)
                        print(AUTH_REQUIRED_MESSAGE, file=sys.stderr)
                        return 2
                    summary = _run_strava_downloader(
                        transport,
                        args,
                        authentication_probe=lambda: check_authentication(
                            transport.context
                        ).authenticated,
                    )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Download error: {exc}", file=sys.stderr)
            return 2
        print(format_download_summary(summary))
        if summary.authentication_required:
            print(AUTH_REQUIRED_MESSAGE)
        return 1 if summary.failed else 0
    if args.command == "strava-reconcile":
        try:
            report = reconcile(args.manifest, args.downloads, args.database)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Reconciliation error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2) if args.json else format_reconciliation(report))
        discrepancies = (
            len(report["missing_download_ids"])
            + len(report["failed_download_ids"])
            + len(report["downloaded_not_imported_ids"])
            + len(report["imported_not_in_manifest_ids"])
            + len(report["duplicate_download_ids"])
            + len(report["download_ids_not_in_manifest"])
            + len(report["invalid_files"])
            + int(report["database_records_without_source_id"])
            + int(report["database_import_failures"])
        )
        return 1 if discrepancies else 0
    return 2


def _run_strava_downloader(session, args, authentication_probe=None):
    downloader = StravaDownloader(
        session,
        delay_seconds=args.delay,
        max_retries=args.retries,
        backoff_seconds=args.backoff,
        timeout_seconds=args.timeout,
        authentication_probe=authentication_probe,
    )
    return downloader.run(
        args.manifest,
        args.output,
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
    )


def _format_plan_reconciliation(report: dict) -> str:
    return "\n".join(
        [
            "PLAN RECONCILIATION",
            f"Planned sessions: {report['planned_sessions']}",
            f"Matched: {report['matched']}",
            f"Probable: {report['probable']}",
            f"Ambiguous: {report['ambiguous']}",
            f"Unmatched: {report['unmatched']}",
            f"Unmatched and due: {report['unmatched_due']}",
            f"Upcoming: {report['upcoming']}",
            f"Metadata completions: {report['metadata_completions']}",
            f"Sessions without planned duration: {len(report['sessions_without_duration'])}",
            f"Low-confidence targets: {len(report['low_confidence_targets'])}",
        ]
    )


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _format_athlete_state(payload: dict) -> str:
    recent = payload["recent_training"]
    load = payload["load"]
    recovery = payload["recovery_context"]
    lines = [
        f"ATHLETE STATE — {payload['date']}",
        f"7-day training: {recent['7d']['sessions']} sessions, {recent['7d']['duration_seconds'] / 3600:.1f} h",
        f"28-day training: {recent['28d']['sessions']} sessions, {recent['28d']['duration_seconds'] / 3600:.1f} h",
        f"7-day load: {load['load_7d']:.0f}",
        "Load/reference ratio: " + (f"{load['recent_to_reference_ratio']:.2f}" if load['recent_to_reference_ratio'] is not None else "unavailable"),
        f"Consecutive training days: {recovery['consecutive_training_days']}",
        "",
        "CONTEXT",
    ]
    for name, item in payload["dimensions"].items():
        lines.append(f"{name.replace('_', ' ').title()}: {item['state'].upper()}")
        lines.extend(f"  - {reason}" for reason in item["reasons"])
    return "\n".join(lines)


def _format_prescription(payload: dict) -> str:
    if payload["action"] == "NO_PLAN":
        return f"COACH — {payload['date']}\nNo explicit planned session."
    workout = payload["prescribed"]
    lines = [
        f"COACH — {payload['date']}",
        f"{workout['sport'].title()} — {workout['title']}",
        f"Decision: {payload['action']}",
        f"Duration: {_duration_text(workout.get('duration_seconds'))}",
    ]
    if workout.get("intensity"):
        lines.append(f"Intensity: {workout['intensity']}")
    if workout.get("targets"):
        lines.append("Targets: " + "; ".join(str(value) for value in workout["targets"].values()))
    lines.extend(["", "Why:"] + [f"- {reason}" for reason in payload["reasons"]])
    lines.append(f"Confidence: {payload['confidence']}")
    lines.append("Rules: " + ", ".join(payload["rules_triggered"]))
    return "\n".join(lines)


def _format_week(results: list[dict], value: date) -> str:
    monday = value - timedelta(days=value.weekday())
    lines = [f"COACHING WEEK — {monday} to {monday + timedelta(days=6)}"]
    if not results:
        lines.append("No explicit planned sessions.")
    for item in results:
        lines.extend(["", _format_prescription(item)])
    return "\n".join(lines)


def _format_evaluation(payload: dict) -> str:
    lines = [
        f"ACTIVITY EVALUATION — {payload['activity_id']}",
        f"Execution: {payload['execution_status']}",
        f"Confidence: {payload['confidence']}",
    ]
    lines.extend(f"{name.replace('_', ' ').title()}: {value}" for name, value in payload["dimensions"].items())
    lines.extend(["", "Evidence:"] + [f"- {value}" for value in payload["evidence"]])
    return "\n".join(lines)


def _format_backtest(payload: dict) -> str:
    lines = [
        "COACHING BACKTEST",
        f"Period: {payload['date_from']} to {payload['date_to']}",
        f"Days evaluated: {payload['days_evaluated']}",
        f"Planned sessions evaluated: {payload['planned_sessions_evaluated']}",
        "Decisions: " + ", ".join(f"{key} {value}" for key, value in sorted(payload["decision_distribution"].items())),
        "Rules: " + ", ".join(f"{key} {value}" for key, value in sorted(payload["rule_frequency"].items())),
    ]
    if payload["suspicious_behavior"]:
        lines.append("Suspicious behavior:")
        lines.extend(f"- {item}" for item in payload["suspicious_behavior"])
    else:
        lines.append("Suspicious behavior: none detected by distribution checks")
    return "\n".join(lines)


def _format_sync(payload: dict) -> str:
    discovery = payload.get("discovery") or {}
    downloads = payload.get("downloads") or {}
    imports = payload.get("imports") or {}
    reconciliation = payload.get("reconciliation") or {}
    coaching = payload.get("coaching") or {}
    calendar = payload.get("calendar") or {}
    quality = payload.get("quality") or {}
    auth_text = "AUTH_REQUIRED" if payload.get("error_code") == "AUTH_REQUIRED" else "OK"
    lines = [
        "ENDURANCE LAB SYNC",
        "==================",
        "",
        "Strava authentication",
        f"    {auth_text}",
        "",
        "Discovery",
        f"    Window: {payload.get('window_from') or 'all history'} to {payload.get('window_to')}",
        f"    Found: {discovery.get('found', 0)}",
        f"    New: {discovery.get('new', 0)}",
        f"    Updated: {discovery.get('updated', 0)}",
        f"    Pages checked: {discovery.get('pages_checked', 0)}",
        "",
        "Downloads",
        f"    Downloaded: {downloads.get('downloaded', 0)}",
        f"    Existing: {downloads.get('already_present', 0)}",
        f"    TCX fallback: {downloads.get('tcx_fallback', 0)}",
        f"    Failed: {downloads.get('failed', 0)}",
        "",
        "Import",
        f"    New activities: {imports.get('new', 0)}",
        f"    Upgraded: {imports.get('upgraded', 0)}",
        f"    Duplicates: {imports.get('duplicates', 0)}",
        f"    Metadata-only: {imports.get('metadata_only', 0)}",
        f"    Errors: {imports.get('errors', 0)}",
        "",
        "Quality",
        f"    Activities checked: {quality.get('total_activities', 0)}",
        f"    Activities with warnings: {quality.get('suspicious_activities', 0)}",
        "",
        "Plan reconciliation",
        f"    Newly matched: {reconciliation.get('newly_matched', 0)}",
        "",
        "Coaching",
        f"    Session costs updated: {coaching.get('session_costs_updated', 0)}",
        f"    Prescriptions evaluated: {coaching.get('prescriptions_evaluated', 0)}",
        f"    Prescriptions updated: {coaching.get('prescriptions_updated', 0)}",
        "",
        "Google Calendar",
        f"    Created: {calendar.get('created', 0)}",
        f"    Updated: {calendar.get('updated', 0)}",
        f"    Unchanged: {calendar.get('unchanged', 0)}",
        f"    Skipped: {'yes' if calendar.get('skipped') else 'no'}",
        "",
        f"Result: {str(payload.get('status', 'unknown')).upper()}",
    ]
    if payload.get("error_summary"):
        lines.extend([f"Reason: {payload['error_summary']}"])
    if payload.get("error_code") == "AUTH_REQUIRED":
        lines.extend(["", AUTH_REQUIRED_MESSAGE, "", "Then run:", "python -m endurance_lab sync"])
    elif payload.get("status") == "success":
        lines.append("Dashboard database is current.")
    return "\n".join(lines)


def _format_calendar_sync(payload: dict) -> str:
    return "\n".join([
        "GOOGLE CALENDAR SYNC",
        f"Calendar: {payload['calendar_id']}",
        f"Window: {payload['date_from']} to {payload['date_to']}",
        f"Prescriptions considered: {payload['considered']}",
        f"Created: {payload['created']}",
        f"Updated: {payload['updated']}",
        f"Unchanged: {payload['unchanged']}",
        f"Removed: {payload['removed']}",
        f"Errors: {payload['errors']}",
        f"Dry run: {'yes' if payload['dry_run'] else 'no'}",
    ])


def _format_activity_calendar_sync(payload: dict) -> str:
    return "\n".join([
        "GOOGLE CALENDAR ACTIVITY SYNC",
        f"Calendar: {payload['calendar_id']}",
        f"Window: {payload['date_from']} to {payload['date_to']}",
        f"Activities considered: {payload['considered']}",
        f"Created: {payload['created']}",
        f"Updated: {payload['updated']}",
        f"Unchanged: {payload['unchanged']}",
        f"Errors: {payload['errors']}",
        f"Dry run: {'yes' if payload['dry_run'] else 'no'}",
    ])


def _format_calendar_status(payload: dict) -> str:
    return "\n".join([
        "GOOGLE CALENDAR STATUS",
        f"Enabled: {'yes' if payload['enabled'] else 'no'}",
        f"OAuth client configured: {'yes' if payload['client_configured'] else 'no'}",
        f"Authorized: {'yes' if payload['authorized'] else 'no'}",
        f"Planned calendar: {payload['planned_calendar_name']} ({payload['calendar_id']})",
        f"Completed calendar: {payload['actual_calendar_name']} ({payload['actual_calendar_id'] or 'not configured'})",
        f"Active planned events: {payload['active_events']}",
        f"Active completed events: {payload['active_activity_events']}",
        f"Planned last synchronized: {payload['last_synced'] or 'never'}",
        f"Completed last synchronized: {payload['activities_last_synced'] or 'never'}",
    ])


def _format_automation_status(payload: dict) -> str:
    phone = payload["phone_access"]
    return "\n".join([
        "ENDURANCE LAB AUTOMATION",
        f"Sync task installed: {'yes' if payload['sync_task'] else 'no'}",
        f"Dashboard task installed: {'yes' if payload['dashboard_task'] else 'no'}",
        f"Runtime configured: {'yes' if payload['runtime_configured'] else 'no'}",
        f"Phone access method: {phone.get('method', 'unavailable')}",
        f"Phone access configured: {'yes' if phone.get('configured') else 'no'}",
        f"Phone access connected: {'yes' if phone['connected'] else 'no'}",
        f"Phone dashboard: {phone['dashboard_url'] or 'unavailable'}",
        phone["message"],
    ])


def _format_phone_access(payload: dict) -> str:
    return "\n".join([
        "PRIVATE PHONE ACCESS",
        f"Method: {payload.get('method', 'unavailable')}",
        f"Configured: {'yes' if payload.get('configured') else 'no'}",
        f"Connected: {'yes' if payload['connected'] else 'no'}",
        f"Dashboard: {payload['dashboard_url'] or 'unavailable'}",
        payload["message"],
    ])


def _format_discovery(payload: dict) -> str:
    return "\n".join([
        "STRAVA DISCOVERY",
        f"Window: {payload.get('from') or 'all history'} to {payload['to']}",
        f"Pages checked: {payload['pages_checked']} / {payload.get('total_pages') or '?'}",
        f"Found: {payload['found']}",
        f"New: {payload['new']}",
        f"Metadata updated: {payload['updated']}",
        f"Existing unchanged: {payload['existing']}",
        f"Complete: {'yes' if payload['complete'] else 'no'}",
    ])


def _format_sync_status(payload: dict) -> str:
    run = payload.get("last_run") or {}
    latest = payload.get("latest_activity") or {}
    return "\n".join([
        "ENDURANCE LAB STATUS",
        "--------------------",
        f"Database: {payload['database']}",
        f"Strava authentication: {payload['authentication']}",
        f"Last sync: {run.get('finished_at') or run.get('started_at') or 'never'}",
        f"Last status: {run.get('status') or 'not run'}",
        f"Last successful sync: {payload.get('last_successful_sync') or 'never'}",
        f"Latest DB activity: {latest.get('started_at') or 'unavailable'} ({latest.get('source_activity_id') or 'no Strava ID'})",
        f"Pending downloads: {payload['pending_downloads']}",
        f"Pending imports: {payload['pending_imports']}",
        f"Metadata-only activities: {payload['metadata_only']}",
    ])


def _format_classification(payload: dict) -> str:
    lines = [
        f"ACTIVITY CLASSIFICATION — {payload['activity_id']}",
        f"Class: {payload['classification']}",
        f"Systemic stress: {payload['stress_level']}",
        f"Muscular load: {payload['muscular_load']}",
        f"Confidence: {payload['confidence']}",
        "",
        "Evidence:",
    ]
    lines.extend(f"- {value}" for value in payload["reasons"])
    return "\n".join(lines)


def _format_session_cost(payload: dict) -> str:
    lines = [
        f"SESSION COST — {payload['activity_id']}",
        f"Overall: {payload['cost_class']}",
        f"Systemic: {payload['systemic_cost']}",
        f"Cardiovascular: {payload['cardiovascular_cost']}",
        f"Muscular: {payload['muscular_cost']} ({payload['muscle_load']})",
        f"Sport-specific: {payload['sport_specific_cost']}",
        f"Intensity / duration: {payload['intensity_cost']} / {payload['duration_cost']}",
        f"Confidence: {payload['confidence']}",
        "",
        "Evidence:",
    ]
    lines.extend(f"- {value}" for value in payload["evidence"])
    if payload.get("context_flags"):
        lines.append("Context: " + ", ".join(payload["context_flags"]))
    return "\n".join(lines)


def _format_coaching_reconciliation(payload: dict) -> str:
    lines = [
        f"COACHING RECONCILIATION — {payload['week_start']}",
        f"Evidence cut-off: {payload['as_of']}",
    ]
    for item in payload["prescriptions"]:
        original = item.get("original") or {}
        lines.append(
            f"{item['date']} | {original.get('priority', '—')} | "
            f"{original.get('title', 'No plan')} -> {item['action']}"
        )
        lines.extend(f"  - {reason}" for reason in item["reasons"])
        if item.get("optional_gate"):
            lines.append(f"  - Gate: {item['optional_gate']['state']}")
    return "\n".join(lines)


def _format_coaching_explanation(payload: list[dict]) -> str:
    if not payload:
        return "No planned coaching decision for this date."
    lines: list[str] = []
    for item in payload:
        lines.append(_format_prescription(item))
        trace = item.get("decision_trace") or {}
        if trace:
            plan = trace.get("planned_session") or {}
            lines.extend([
                "", "Decision trace:",
                f"- Priority: {plan.get('priority', '—')} ({plan.get('priority_source', 'inferred')})",
                f"- Recovery runway: {trace.get('recovery_runway_h') or 'unavailable'} h",
            ])
            actual = trace.get("actual_context") or []
            if actual:
                lines.append("- Preceding actual training:")
                for value in actual[-5:]:
                    cost = value.get("session_cost") or {}
                    lines.append(
                        f"  - {value['date']} {value.get('name') or value.get('source_activity_id') or value['activity_id']}: "
                        f"{cost.get('systemic_cost', 'unknown')} systemic, "
                        f"{cost.get('muscular_cost', 'unknown')} {cost.get('muscle_load', '')} muscular"
                    )
            for flag in trace.get("interference_flags") or []:
                lines.append(f"- Interference: {flag['type']} ({flag['hours_before']} h; {flag['evidence']})")
            for protected in trace.get("protected_sessions") or []:
                lines.append(f"- Protected: {protected['date']} {protected['title']} ({protected['priority']})")
            if item.get("optional_gate"):
                gate = item["optional_gate"]
                lines.append(
                    f"- Optional gate: {gate['state']}; proceed only for "
                    f"{', '.join(gate['acceptable_states'])}, otherwise {gate['fallback']['title']}"
                )
    return "\n\n".join(lines)


def _duration_text(seconds) -> str:
    return "unavailable" if seconds is None else f"{float(seconds) / 60:.0f} min"
if __name__ == "__main__":
    sys.exit(main())
