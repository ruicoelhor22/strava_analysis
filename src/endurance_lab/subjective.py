from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from endurance_lab.db import connect, init_db


FIELDS = ("sleep_quality", "fatigue", "leg_soreness", "stress", "motivation", "recovery_rpe")


def save_checkin(
    checkin_date: date,
    database: str | Path | None = None,
    notes: str | None = None,
    **values: int | None,
) -> dict[str, Any]:
    init_db(database)
    cleaned = {field: _score(values.get(field), field) for field in FIELDS}
    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        connection.execute(
            """INSERT INTO daily_checkins (
                   checkin_date, sleep_quality, fatigue, leg_soreness, stress,
                   motivation, recovery_rpe, notes, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(checkin_date) DO UPDATE SET
                   sleep_quality = excluded.sleep_quality, fatigue = excluded.fatigue,
                   leg_soreness = excluded.leg_soreness, stress = excluded.stress,
                   motivation = excluded.motivation, recovery_rpe = excluded.recovery_rpe,
                   notes = excluded.notes, updated_at = excluded.updated_at""",
            (
                checkin_date.isoformat(), *(cleaned[field] for field in FIELDS),
                (notes or "").strip() or None, now, now,
            ),
        )
        connection.commit()
    return get_checkin(checkin_date, database) or {}


def get_checkin(checkin_date: date, database: str | Path | None = None) -> dict[str, Any] | None:
    init_db(database)
    with connect(database) as connection:
        row = connection.execute(
            "SELECT * FROM daily_checkins WHERE checkin_date = ?", (checkin_date.isoformat(),)
        ).fetchone()
    return dict(row) if row else None


def latest_checkin(as_of: date, database: str | Path | None = None) -> dict[str, Any] | None:
    init_db(database)
    with connect(database) as connection:
        row = connection.execute(
            """SELECT * FROM daily_checkins WHERE checkin_date <= ?
               ORDER BY checkin_date DESC LIMIT 1""",
            (as_of.isoformat(),),
        ).fetchone()
    return dict(row) if row else None


def _score(value: int | None, field: str) -> int | None:
    if value is None:
        return None
    score = int(value)
    if not 1 <= score <= 5:
        raise ValueError(f"{field} must be between 1 and 5")
    return score
