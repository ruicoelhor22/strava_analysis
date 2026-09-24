from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from endurance_lab.db import init_db, transaction
from endurance_lab.plan_models import PlanImportSummary
from endurance_lab.plan_parser import parse_workbook


def import_training_plan(
    workbook: str | Path | None = None,
    database: str | Path | None = None,
) -> PlanImportSummary:
    parsed = parse_workbook(workbook)
    init_db(database)
    now = datetime.now(timezone.utc).isoformat()
    sports = dict(sorted(Counter(item.sport for item in parsed.sessions).items()))
    warnings = _warnings(parsed)
    with transaction(database) as connection:
        existing = connection.execute(
            "SELECT id, source_sha256 FROM training_plans WHERE source_path = ?",
            (str(parsed.source),),
        ).fetchone()
        if existing and str(existing["source_sha256"]) == parsed.source_sha256:
            return PlanImportSummary(
                parsed.source, True, len(parsed.sheets), len(parsed.phases), len(parsed.weeks),
                len(parsed.sessions), parsed.date_start, parsed.date_end, sports, warnings,
            )
        if existing:
            plan_id = int(existing["id"])
            connection.execute(
                """UPDATE training_plans SET name = ?, source_filename = ?, source_sha256 = ?,
                          workbook_modified_at = ?, imported_at = ?, date_start = ?, date_end = ?,
                          active = 1, metadata_json = ? WHERE id = ?""",
                _plan_values(parsed, now) + (plan_id,),
            )
        else:
            cursor = connection.execute(
                """INSERT INTO training_plans (
                       name, source_path, source_filename, source_sha256,
                       workbook_modified_at, imported_at, date_start, date_end, active, metadata_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (parsed.name, str(parsed.source), parsed.source.name, parsed.source_sha256,
                 datetime.fromtimestamp(parsed.source.stat().st_mtime, timezone.utc).isoformat(),
                 now, parsed.date_start.isoformat(), parsed.date_end.isoformat(),
                 json.dumps({"sheets": list(parsed.sheets)}, ensure_ascii=False)),
            )
            plan_id = int(cursor.lastrowid)

        connection.execute(
            """INSERT OR IGNORE INTO training_plan_sources (
                   plan_id, source_sha256, source_path, imported_at, sheet_names_json, workbook_json
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (plan_id, parsed.source_sha256, str(parsed.source), now,
             json.dumps(list(parsed.sheets), ensure_ascii=False),
             json.dumps(parsed.workbook_snapshot, ensure_ascii=False)),
        )
        connection.execute("UPDATE training_phases SET active = 0 WHERE plan_id = ?", (plan_id,))
        connection.execute("UPDATE training_weeks SET active = 0 WHERE plan_id = ?", (plan_id,))
        connection.execute("UPDATE planned_sessions SET active = 0 WHERE plan_id = ?", (plan_id,))

        phase_ids: dict[str, int] = {}
        for phase in parsed.phases:
            connection.execute(
                """INSERT INTO training_phases (
                       plan_id, name, start_date, end_date, objectives, source_sheet,
                       source_row, raw_json, active
                   ) VALUES (?, ?, ?, ?, ?, 'Training Calendar', NULL, ?, 1)
                   ON CONFLICT(plan_id, name, start_date) DO UPDATE SET
                       end_date = excluded.end_date, objectives = excluded.objectives,
                       raw_json = excluded.raw_json, active = 1""",
                (plan_id, phase.name, phase.start_date.isoformat(), phase.end_date.isoformat(),
                 phase.objectives, json.dumps(phase.raw_source, ensure_ascii=False)),
            )
            phase_ids[phase.name] = int(connection.execute(
                "SELECT id FROM training_phases WHERE plan_id = ? AND name = ? AND start_date = ?",
                (plan_id, phase.name, phase.start_date.isoformat()),
            ).fetchone()[0])

        week_ids: dict[str, int] = {}
        for week in parsed.weeks:
            connection.execute(
                """INSERT INTO training_weeks (
                       plan_id, phase_id, week_start, week_end, phase_name, week_type,
                       objectives, raw_json, active
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                   ON CONFLICT(plan_id, week_start) DO UPDATE SET
                       phase_id = excluded.phase_id, week_end = excluded.week_end,
                       phase_name = excluded.phase_name, week_type = excluded.week_type,
                       objectives = excluded.objectives, raw_json = excluded.raw_json, active = 1""",
                (plan_id, phase_ids.get(week.phase), week.week_start.isoformat(),
                 week.week_end.isoformat(), week.phase, week.week_type, week.objectives,
                 json.dumps(week.raw_source, ensure_ascii=False)),
            )
            week_ids[week.week_start.isoformat()] = int(connection.execute(
                "SELECT id FROM training_weeks WHERE plan_id = ? AND week_start = ?",
                (plan_id, week.week_start.isoformat()),
            ).fetchone()[0])

        for session in parsed.sessions:
            values = (
                plan_id, week_ids.get(session.week_start.isoformat()), session.planned_date.isoformat(),
                session.sport, session.session_type, session.workout_code, session.title,
                session.description, session.planned_duration_seconds, session.planned_distance_m,
                None, session.intensity, session.interval_structure, session.priority, session.notes,
                "workbook" if session.priority else "inferred",
                session.source_status, json.dumps(session.workbook_actual, ensure_ascii=False),
                "Training Calendar", session.source_row, f"A{session.source_row}:AI{session.source_row}",
                session.source_key, json.dumps(session.raw_source, ensure_ascii=False), now,
            )
            connection.execute(
                """INSERT INTO planned_sessions (
                       plan_id, week_id, planned_date, sport, session_type, workout_code,
                       title, description, planned_duration_seconds, planned_distance_m,
                       planned_load, intensity, interval_structure, priority, notes, priority_source,
                       source_status, workbook_actual_json, source_sheet, source_row,
                       source_range, source_key, raw_source_text, active, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                   ON CONFLICT(plan_id, source_key) DO UPDATE SET
                       week_id = excluded.week_id, planned_date = excluded.planned_date,
                       sport = excluded.sport, session_type = excluded.session_type,
                       workout_code = excluded.workout_code, title = excluded.title,
                       description = excluded.description,
                       planned_duration_seconds = excluded.planned_duration_seconds,
                       planned_distance_m = excluded.planned_distance_m,
                       planned_load = excluded.planned_load, intensity = excluded.intensity,
                       interval_structure = excluded.interval_structure,
                       priority = excluded.priority, notes = excluded.notes,
                       priority_source = excluded.priority_source,
                       source_status = excluded.source_status,
                       workbook_actual_json = excluded.workbook_actual_json,
                       source_row = excluded.source_row, source_range = excluded.source_range,
                       raw_source_text = excluded.raw_source_text, active = 1,
                       updated_at = excluded.updated_at""",
                values,
            )
            session_id = int(connection.execute(
                "SELECT id FROM planned_sessions WHERE plan_id = ? AND source_key = ?",
                (plan_id, session.source_key),
            ).fetchone()[0])
            connection.execute(
                "DELETE FROM planned_session_targets WHERE planned_session_id = ?", (session_id,)
            )
            for target in session.targets:
                connection.execute(
                    """INSERT INTO planned_session_targets (
                           planned_session_id, target_type, raw_text, minimum_value,
                           maximum_value, unit, confidence
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, target.target_type, target.raw_text, target.minimum_value,
                     target.maximum_value, target.unit, target.confidence),
                )
            targets_json = json.dumps(
                {target.target_type: target.raw_text for target in session.targets}, ensure_ascii=False
            )
            connection.execute(
                """INSERT INTO workout_prescriptions (
                       planned_session_id, prescribed_date, sport, title, description,
                       duration_seconds, distance_m, intensity, targets_json, reason,
                       status, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Mirrors imported coaching plan', 'mirrored', ?)
                   ON CONFLICT(planned_session_id, status) DO UPDATE SET
                       prescribed_date = excluded.prescribed_date, sport = excluded.sport,
                       title = excluded.title, description = excluded.description,
                       duration_seconds = excluded.duration_seconds, distance_m = excluded.distance_m,
                       intensity = excluded.intensity, targets_json = excluded.targets_json,
                       reason = excluded.reason""",
                (session_id, session.planned_date.isoformat(), session.sport, session.title,
                 session.description, session.planned_duration_seconds, session.planned_distance_m,
                 session.intensity, targets_json, now),
            )

    return PlanImportSummary(
        parsed.source, False, len(parsed.sheets), len(parsed.phases), len(parsed.weeks),
        len(parsed.sessions), parsed.date_start, parsed.date_end, sports, warnings,
    )


def _plan_values(parsed, now: str) -> tuple:
    return (
        parsed.name, parsed.source.name, parsed.source_sha256,
        datetime.fromtimestamp(parsed.source.stat().st_mtime, timezone.utc).isoformat(),
        now, parsed.date_start.isoformat(), parsed.date_end.isoformat(),
        json.dumps({"sheets": list(parsed.sheets)}, ensure_ascii=False),
    )


def _warnings(parsed) -> tuple[str, ...]:
    warnings = []
    if any(item.planned_duration_seconds is None for item in parsed.sessions):
        warnings.append("Some planned sessions have no explicit duration; duration adherence remains unavailable.")
    if not any(item.planned_distance_m is not None for item in parsed.sessions):
        warnings.append("The calendar has no dedicated planned-distance column; distance is only inferred from explicit titles.")
    warnings.append("Non-calendar workbook sheets are preserved as raw provenance and are not silently interpreted as sessions.")
    return tuple(warnings)


def format_import_summary(summary: PlanImportSummary) -> str:
    lines = [
        "TRAINING PLAN IMPORT",
        f"Workbook: {summary.workbook}",
        f"Workbook unchanged: {'yes' if summary.unchanged else 'no'}",
        f"Sheets inspected: {summary.sheets_inspected}",
        f"Training phases: {summary.phases}",
        f"Training weeks: {summary.weeks}",
        f"Planned sessions: {summary.sessions}",
        f"Date range: {summary.date_start} to {summary.date_end}",
        "Sports: " + ", ".join(f"{key} {value}" for key, value in summary.sports.items()),
    ]
    lines.extend(f"Warning: {warning}" for warning in summary.warnings)
    return "\n".join(lines)
