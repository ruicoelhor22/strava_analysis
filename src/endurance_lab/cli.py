from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from endurance_lab.analytics import analyze_database
from endurance_lab.config import ensure_local_layout, paths
from endurance_lab.db import init_db
from endurance_lab.importer import SUPPORTED_FORMATS, discover_files, import_path
from endurance_lab.quality import ingestion_quality_report
from endurance_lab.queries import data_quality
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


if __name__ == "__main__":
    sys.exit(main())
