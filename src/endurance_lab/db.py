from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from endurance_lab.config import paths


SCHEMA_VERSION = 2

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
    connection.execute(
        """INSERT OR IGNORE INTO activity_sources (
               activity_id, import_id, source_format, source_filename, raw_path,
               sha256, source_activity_id, quality_score, selected, metadata_json, imported_at
           )
           SELECT a.id, a.import_id, 'tcx', a.source_filename, a.raw_path,
                  i.sha256, a.source_activity_id, 0, 1, NULL, a.imported_at
           FROM activities a JOIN imports i ON i.id = a.import_id"""
    )


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
