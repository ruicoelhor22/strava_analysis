from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from endurance_lab.config import paths


SCHEMA_VERSION = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY,
    source_filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    source_format TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual_file',
    raw_path TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('processing', 'imported', 'failed')),
    activity_count INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS activities (
    id INTEGER PRIMARY KEY,
    stable_id TEXT NOT NULL UNIQUE,
    source_activity_id TEXT,
    source_filename TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    import_id INTEGER NOT NULL REFERENCES imports(id),
    source_format TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual_file',
    identity_key TEXT,
    source_metadata_json TEXT,
    quality_score INTEGER NOT NULL DEFAULT 0,
    metadata_only INTEGER NOT NULL DEFAULT 0,
    sport TEXT NOT NULL,
    name TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    elapsed_seconds REAL,
    moving_seconds REAL,
    distance_m REAL,
    ascent_m REAL,
    descent_m REAL,
    calories REAL,
    avg_hr REAL,
    max_hr REAL,
    avg_speed_mps REAL,
    max_speed_mps REAL,
    avg_cadence REAL,
    max_cadence REAL,
    avg_power_w REAL,
    max_power_w REAL,
    device TEXT,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_laps (
    id INTEGER PRIMARY KEY,
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    lap_number INTEGER NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    duration_seconds REAL,
    distance_m REAL,
    ascent_m REAL,
    descent_m REAL,
    calories REAL,
    avg_hr REAL,
    max_hr REAL,
    avg_speed_mps REAL,
    max_speed_mps REAL,
    avg_cadence REAL,
    max_cadence REAL,
    avg_power_w REAL,
    max_power_w REAL,
    UNIQUE(activity_id, lap_number)
);

CREATE TABLE IF NOT EXISTS activity_streams (
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    recorded_at TEXT,
    elapsed_seconds REAL,
    distance_m REAL,
    latitude REAL,
    longitude REAL,
    altitude_m REAL,
    heart_rate REAL,
    cadence REAL,
    speed_mps REAL,
    power_w REAL,
    temperature_c REAL,
    moving INTEGER,
    PRIMARY KEY(activity_id, sequence)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS activity_sources (
    id INTEGER PRIMARY KEY,
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    import_id INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    source_format TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    source_activity_id TEXT,
    quality_score INTEGER NOT NULL,
    selected INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    imported_at TEXT NOT NULL,
    UNIQUE(activity_id, import_id)
);

CREATE TABLE IF NOT EXISTS strength_sets (
    id INTEGER PRIMARY KEY,
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    exercise_name TEXT NOT NULL,
    set_number INTEGER NOT NULL,
    repetitions INTEGER,
    load_value REAL,
    load_unit TEXT,
    started_at TEXT,
    duration_seconds REAL,
    metadata_json TEXT,
    UNIQUE(activity_id, exercise_name, set_number)
);

CREATE TABLE IF NOT EXISTS derived_activity_metrics (
    activity_id INTEGER PRIMARY KEY REFERENCES activities(id) ON DELETE CASCADE,
    metric_version INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    ftp_w REAL,
    normalized_power_w REAL,
    intensity_factor REAL,
    estimated_tss REAL,
    estimated_hr_load REAL,
    selected_load REAL,
    load_method TEXT,
    efficiency_factor REAL,
    aerobic_decoupling_pct REAL,
    decoupling_status TEXT,
    decoupling_reason TEXT,
    first_half_output REAL,
    second_half_output REAL,
    first_half_hr REAL,
    second_half_hr REAL,
    late_fade_pct REAL,
    pace_seconds_per_km REAL,
    hr_coverage REAL,
    moving_ratio REAL,
    hr_zones_json TEXT,
    power_zones_json TEXT,
    best_power_json TEXT
);

CREATE TABLE IF NOT EXISTS power_curve_results (
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    duration_seconds INTEGER NOT NULL,
    best_power_w REAL NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY(activity_id, duration_seconds)
);

CREATE TABLE IF NOT EXISTS daily_training_load (
    day TEXT PRIMARY KEY,
    total_load REAL NOT NULL,
    cycling_load REAL NOT NULL DEFAULT 0,
    running_load REAL NOT NULL DEFAULT 0,
    swimming_load REAL NOT NULL DEFAULT 0,
    strength_load REAL NOT NULL DEFAULT 0,
    other_load REAL NOT NULL DEFAULT 0,
    duration_seconds REAL NOT NULL DEFAULT 0,
    sessions INTEGER NOT NULL DEFAULT 0,
    hard_sessions INTEGER NOT NULL DEFAULT 0,
    long_sessions INTEGER NOT NULL DEFAULT 0,
    fitness REAL NOT NULL,
    fatigue REAL NOT NULL,
    form REAL NOT NULL,
    computed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_plans (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    source_filename TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    workbook_modified_at TEXT,
    imported_at TEXT NOT NULL,
    date_start TEXT,
    date_end TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS training_plan_sources (
    id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES training_plans(id) ON DELETE CASCADE,
    source_sha256 TEXT NOT NULL,
    source_path TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    sheet_names_json TEXT NOT NULL,
    workbook_json TEXT NOT NULL,
    UNIQUE(plan_id, source_sha256)
);

CREATE TABLE IF NOT EXISTS training_phases (
    id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES training_plans(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    objectives TEXT,
    source_sheet TEXT NOT NULL,
    source_row INTEGER,
    raw_json TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(plan_id, name, start_date)
);

CREATE TABLE IF NOT EXISTS training_weeks (
    id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES training_plans(id) ON DELETE CASCADE,
    phase_id INTEGER REFERENCES training_phases(id) ON DELETE SET NULL,
    week_start TEXT NOT NULL,
    week_end TEXT NOT NULL,
    phase_name TEXT,
    week_type TEXT,
    objectives TEXT,
    raw_json TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(plan_id, week_start)
);

CREATE TABLE IF NOT EXISTS planned_sessions (
    id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES training_plans(id) ON DELETE CASCADE,
    week_id INTEGER REFERENCES training_weeks(id) ON DELETE SET NULL,
    planned_date TEXT NOT NULL,
    sport TEXT NOT NULL,
    session_type TEXT,
    workout_code TEXT,
    title TEXT NOT NULL,
    description TEXT,
    planned_duration_seconds REAL,
    planned_distance_m REAL,
    planned_load REAL,
    intensity TEXT,
    interval_structure TEXT,
    priority TEXT,
    priority_source TEXT NOT NULL DEFAULT 'inferred',
    notes TEXT,
    source_status TEXT,
    workbook_actual_json TEXT,
    source_sheet TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    source_range TEXT,
    source_key TEXT NOT NULL,
    raw_source_text TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    UNIQUE(plan_id, source_key)
);

CREATE TABLE IF NOT EXISTS planned_session_targets (
    id INTEGER PRIMARY KEY,
    planned_session_id INTEGER NOT NULL REFERENCES planned_sessions(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    minimum_value REAL,
    maximum_value REAL,
    unit TEXT,
    confidence TEXT NOT NULL DEFAULT 'low',
    UNIQUE(planned_session_id, target_type)
);

CREATE TABLE IF NOT EXISTS planned_activity_matches (
    id INTEGER PRIMARY KEY,
    planned_session_id INTEGER NOT NULL REFERENCES planned_sessions(id) ON DELETE CASCADE,
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    match_score REAL NOT NULL,
    match_method TEXT NOT NULL,
    match_status TEXT NOT NULL CHECK(match_status IN ('matched', 'probable', 'ambiguous', 'manual')),
    reason_json TEXT NOT NULL,
    is_selected INTEGER NOT NULL DEFAULT 0,
    matched_at TEXT NOT NULL,
    manual_notes TEXT,
    UNIQUE(planned_session_id, activity_id)
);

CREATE TABLE IF NOT EXISTS workout_prescriptions (
    id INTEGER PRIMARY KEY,
    planned_session_id INTEGER REFERENCES planned_sessions(id) ON DELETE SET NULL,
    prescribed_date TEXT NOT NULL,
    sport TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    duration_seconds REAL,
    distance_m REAL,
    intensity TEXT,
    targets_json TEXT,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'mirrored',
    action TEXT NOT NULL DEFAULT 'KEEP',
    confidence TEXT NOT NULL DEFAULT 'medium',
    original_json TEXT,
    prescribed_json TEXT,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    rules_json TEXT NOT NULL DEFAULT '[]',
    as_of_date TEXT,
    engine_version INTEGER NOT NULL DEFAULT 1,
    optional_gate_json TEXT,
    decision_trace_json TEXT NOT NULL DEFAULT '{}',
    recovery_runway_h REAL,
    protected_sessions_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    superseded_by INTEGER REFERENCES workout_prescriptions(id),
    UNIQUE(planned_session_id, status)
);

CREATE TABLE IF NOT EXISTS activity_classifications (
    activity_id INTEGER PRIMARY KEY REFERENCES activities(id) ON DELETE CASCADE,
    classification TEXT NOT NULL,
    confidence TEXT NOT NULL,
    stress_level TEXT NOT NULL,
    muscular_load TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    model_version INTEGER NOT NULL,
    classified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_costs (
    activity_id INTEGER PRIMARY KEY REFERENCES activities(id) ON DELETE CASCADE,
    cost_class TEXT NOT NULL,
    systemic_cost TEXT NOT NULL,
    cardiovascular_cost TEXT NOT NULL,
    muscular_cost TEXT NOT NULL,
    muscle_load TEXT NOT NULL,
    sport_specific_cost TEXT NOT NULL,
    intensity_cost TEXT NOT NULL,
    duration_cost TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    context_flags_json TEXT NOT NULL DEFAULT '[]',
    exercise_summary_json TEXT NOT NULL DEFAULT '{}',
    model_version INTEGER NOT NULL,
    classified_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_intervals (
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    interval_number INTEGER NOT NULL,
    lap_number INTEGER,
    duration_seconds REAL,
    distance_m REAL,
    avg_power_w REAL,
    avg_hr REAL,
    avg_pace_seconds_per_km REAL,
    avg_cadence REAL,
    target_adherence TEXT,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(activity_id, interval_number)
);

CREATE TABLE IF NOT EXISTS workout_evaluations (
    activity_id INTEGER PRIMARY KEY REFERENCES activities(id) ON DELETE CASCADE,
    planned_session_id INTEGER REFERENCES planned_sessions(id) ON DELETE SET NULL,
    execution_status TEXT NOT NULL,
    confidence TEXT NOT NULL,
    dimensions_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    evaluated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_checkins (
    checkin_date TEXT PRIMARY KEY,
    sleep_quality INTEGER,
    fatigue INTEGER,
    leg_soreness INTEGER,
    stress INTEGER,
    motivation INTEGER,
    recovery_rpe INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_from TEXT,
    requested_to TEXT,
    discovered_count INTEGER NOT NULL DEFAULT 0,
    downloaded_count INTEGER NOT NULL DEFAULT 0,
    metadata_only_count INTEGER NOT NULL DEFAULT 0,
    imported_count INTEGER NOT NULL DEFAULT 0,
    matched_count INTEGER NOT NULL DEFAULT 0,
    prescriptions_updated INTEGER NOT NULL DEFAULT 0,
    warnings_count INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    error_summary TEXT,
    summary_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sync_run_stages (
    id INTEGER PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES sync_runs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    checkpoint_json TEXT NOT NULL DEFAULT '{}',
    error_code TEXT,
    error_summary TEXT,
    UNIQUE(sync_run_id, stage)
);

CREATE TABLE IF NOT EXISTS strava_discovery_runs (
    id INTEGER PRIMARY KEY,
    sync_run_id INTEGER REFERENCES sync_runs(id) ON DELETE SET NULL,
    requested_from TEXT,
    requested_to TEXT,
    current_page INTEGER NOT NULL DEFAULT 0,
    total_pages INTEGER,
    discovered_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_summary TEXT
);

CREATE TABLE IF NOT EXISTS calendar_event_links (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'google',
    calendar_id TEXT NOT NULL,
    planned_session_id INTEGER NOT NULL REFERENCES planned_sessions(id) ON DELETE CASCADE,
    prescription_id INTEGER REFERENCES workout_prescriptions(id) ON DELETE SET NULL,
    external_event_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    event_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_synced_at TEXT NOT NULL,
    error TEXT,
    UNIQUE(provider, calendar_id, planned_session_id)
);

CREATE TABLE IF NOT EXISTS activity_calendar_event_links (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'google',
    calendar_id TEXT NOT NULL,
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    external_event_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    event_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_synced_at TEXT NOT NULL,
    error TEXT,
    UNIQUE(provider, calendar_id, activity_id)
);

CREATE INDEX IF NOT EXISTS idx_activities_started_sport
ON activities(started_at, sport);
CREATE INDEX IF NOT EXISTS idx_streams_activity_elapsed
ON activity_streams(activity_id, elapsed_seconds);
CREATE INDEX IF NOT EXISTS idx_laps_activity
ON activity_laps(activity_id, lap_number);
CREATE INDEX IF NOT EXISTS idx_power_curve_duration
ON power_curve_results(duration_seconds, best_power_w DESC);
CREATE INDEX IF NOT EXISTS idx_activities_source_activity_id
ON activities(source_activity_id) WHERE source_activity_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activities_identity
ON activities(identity_key);
CREATE INDEX IF NOT EXISTS idx_activity_sources_activity
ON activity_sources(activity_id, selected);
CREATE INDEX IF NOT EXISTS idx_strength_sets_activity
ON strength_sets(activity_id, exercise_name);
CREATE INDEX IF NOT EXISTS idx_plan_phases_dates
ON training_phases(plan_id, start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_plan_weeks_start
ON training_weeks(plan_id, week_start);
CREATE INDEX IF NOT EXISTS idx_planned_sessions_date_sport
ON planned_sessions(planned_date, sport, active);
CREATE INDEX IF NOT EXISTS idx_planned_sessions_week
ON planned_sessions(week_id, planned_date);
CREATE INDEX IF NOT EXISTS idx_plan_matches_session_selected
ON planned_activity_matches(planned_session_id, is_selected);
CREATE INDEX IF NOT EXISTS idx_plan_matches_activity
ON planned_activity_matches(activity_id, is_selected);
CREATE INDEX IF NOT EXISTS idx_prescriptions_date
ON workout_prescriptions(prescribed_date, sport, status);
CREATE INDEX IF NOT EXISTS idx_classifications_class_stress
ON activity_classifications(classification, stress_level);
CREATE INDEX IF NOT EXISTS idx_session_costs_class
ON session_costs(cost_class, muscular_cost, muscle_load);
CREATE INDEX IF NOT EXISTS idx_checkins_date
ON daily_checkins(checkin_date);
CREATE INDEX IF NOT EXISTS idx_sync_runs_started
ON sync_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sync_stages_run
ON sync_run_stages(sync_run_id, stage);
CREATE INDEX IF NOT EXISTS idx_calendar_event_date
ON calendar_event_links(provider, calendar_id, event_date, status);
CREATE INDEX IF NOT EXISTS idx_activity_calendar_event_date
ON activity_calendar_event_links(provider, calendar_id, event_date, status);
"""


def connect(database: str | Path | None = None) -> sqlite3.Connection:
    path = Path(database) if database else paths().database
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def init_db(database: str | Path | None = None) -> Path:
    path = Path(database) if database else paths().database
    with connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)")
        row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if row is None:
            connection.executescript(SCHEMA)
            connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
        elif int(row[0]) == 1:
            _migrate_v1_to_v2(connection)
            _migrate_v2_to_v3(connection)
            _migrate_v3_to_v4(connection)
            _migrate_v4_to_v5(connection)
            _migrate_v5_to_v6(connection)
            _migrate_v6_to_v7(connection)
            _migrate_v7_to_v8(connection)
            connection.executescript(SCHEMA)
            connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        elif int(row[0]) == 2:
            _migrate_v2_to_v3(connection)
            _migrate_v3_to_v4(connection)
            _migrate_v4_to_v5(connection)
            _migrate_v5_to_v6(connection)
            _migrate_v6_to_v7(connection)
            _migrate_v7_to_v8(connection)
            connection.executescript(SCHEMA)
            connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        elif int(row[0]) == 3:
            _migrate_v3_to_v4(connection)
            _migrate_v4_to_v5(connection)
            _migrate_v5_to_v6(connection)
            _migrate_v6_to_v7(connection)
            _migrate_v7_to_v8(connection)
            connection.executescript(SCHEMA)
            connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        elif int(row[0]) == 4:
            _migrate_v4_to_v5(connection)
            _migrate_v5_to_v6(connection)
            _migrate_v6_to_v7(connection)
            _migrate_v7_to_v8(connection)
            connection.executescript(SCHEMA)
            connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        elif int(row[0]) == 5:
            _migrate_v5_to_v6(connection)
            _migrate_v6_to_v7(connection)
            _migrate_v7_to_v8(connection)
            connection.executescript(SCHEMA)
            connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        elif int(row[0]) == 6:
            _migrate_v6_to_v7(connection)
            _migrate_v7_to_v8(connection)
            connection.executescript(SCHEMA)
            connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        elif int(row[0]) == 7:
            _migrate_v7_to_v8(connection)
            connection.executescript(SCHEMA)
            connection.execute("UPDATE schema_meta SET version = ?", (SCHEMA_VERSION,))
        elif int(row[0]) == SCHEMA_VERSION:
            connection.executescript(SCHEMA)
        else:
            raise RuntimeError(
                f"Unsupported database schema {row[0]}; expected {SCHEMA_VERSION}."
            )
        connection.execute("PRAGMA optimize")
    return path


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    import_columns = _columns(connection, "imports")
    if "source_format" not in import_columns:
        connection.execute("ALTER TABLE imports ADD COLUMN source_format TEXT NOT NULL DEFAULT 'tcx'")
    if "source" not in import_columns:
        connection.execute("ALTER TABLE imports ADD COLUMN source TEXT NOT NULL DEFAULT 'manual_file'")

    activity_columns = _columns(connection, "activities")
    additions = {
        "source_format": "TEXT NOT NULL DEFAULT 'tcx'",
        "source": "TEXT NOT NULL DEFAULT 'manual_file'",
        "identity_key": "TEXT",
        "source_metadata_json": "TEXT",
        "quality_score": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in additions.items():
        if name not in activity_columns:
            connection.execute(f"ALTER TABLE activities ADD COLUMN {name} {definition}")
    connection.execute("UPDATE activities SET identity_key = stable_id WHERE identity_key IS NULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS activity_sources (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
            import_id INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
            source_format TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            raw_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            source_activity_id TEXT,
            quality_score INTEGER NOT NULL,
            selected INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT,
            imported_at TEXT NOT NULL,
            UNIQUE(activity_id, import_id)
        );
        CREATE TABLE IF NOT EXISTS strength_sets (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
            exercise_name TEXT NOT NULL,
            set_number INTEGER NOT NULL,
            repetitions INTEGER,
            load_value REAL,
            load_unit TEXT,
            started_at TEXT,
            duration_seconds REAL,
            metadata_json TEXT,
            UNIQUE(activity_id, exercise_name, set_number)
        );
        """
    )


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    activity_columns = _columns(connection, "activities")
    if "metadata_only" not in activity_columns:
        connection.execute(
            "ALTER TABLE activities ADD COLUMN metadata_only INTEGER NOT NULL DEFAULT 0"
        )
    connection.execute(
        """INSERT OR IGNORE INTO activity_sources (
               activity_id, import_id, source_format, source_filename, raw_path,
               sha256, source_activity_id, quality_score, selected, metadata_json, imported_at
           )
           SELECT a.id, a.import_id, 'tcx', a.source_filename, a.raw_path,
                  i.sha256, a.source_activity_id, 0, 1, NULL, a.imported_at
           FROM activities a JOIN imports i ON i.id = a.import_id"""
    )


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    prescription_columns = _columns(connection, "workout_prescriptions")
    additions = {
        "action": "TEXT NOT NULL DEFAULT 'KEEP'",
        "confidence": "TEXT NOT NULL DEFAULT 'medium'",
        "original_json": "TEXT",
        "prescribed_json": "TEXT",
        "evidence_json": "TEXT NOT NULL DEFAULT '[]'",
        "rules_json": "TEXT NOT NULL DEFAULT '[]'",
        "as_of_date": "TEXT",
        "engine_version": "INTEGER NOT NULL DEFAULT 1",
    }
    for name, definition in additions.items():
        if name not in prescription_columns:
            connection.execute(
                f"ALTER TABLE workout_prescriptions ADD COLUMN {name} {definition}"
            )


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    planned_columns = _columns(connection, "planned_sessions")
    if planned_columns and "priority_source" not in planned_columns:
        connection.execute(
            "ALTER TABLE planned_sessions ADD COLUMN priority_source TEXT NOT NULL DEFAULT 'inferred'"
        )
        connection.execute(
            "UPDATE planned_sessions SET priority_source = 'workbook' WHERE priority IS NOT NULL AND priority != ''"
        )
    prescription_columns = _columns(connection, "workout_prescriptions")
    additions = {
        "optional_gate_json": "TEXT",
        "decision_trace_json": "TEXT NOT NULL DEFAULT '{}'",
        "recovery_runway_h": "REAL",
        "protected_sessions_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, definition in additions.items():
        if not prescription_columns:
            break
        if name not in prescription_columns:
            connection.execute(
                f"ALTER TABLE workout_prescriptions ADD COLUMN {name} {definition}"
            )


def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
    # Version 6 is additive; CREATE TABLE IF NOT EXISTS in SCHEMA performs the migration.
    return None


def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
    # Version 7 is additive; CREATE TABLE IF NOT EXISTS in SCHEMA performs the migration.
    return None


def _migrate_v7_to_v8(connection: sqlite3.Connection) -> None:
    # Version 8 adds idempotent links for completed-activity calendar events.
    return None


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


@contextmanager
def transaction(database: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
