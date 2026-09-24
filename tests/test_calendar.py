from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from endurance_lab.automation import install_windows_tasks
from endurance_lab.db import connect, init_db
from endurance_lab.google_calendar import (
    calendar_status, configure_google_calendars, sync_activity_calendar, sync_google_calendar,
)


class FakeCalendarClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[tuple[str, dict]] = []
        self.deleted: list[str] = []

    def insert(self, calendar_id, event):
        self.created.append(event)
        return {"id": f"event-{len(self.created)}"}

    def update(self, calendar_id, event_id, event):
        self.updated.append((event_id, event))
        return {"id": event_id}

    def delete(self, calendar_id, event_id):
        self.deleted.append(event_id)


class FakeSetupClient(FakeCalendarClient):
    def list_calendars(self):
        return [{"id": "strava-id", "summary": "Strava"}]

    def create_calendar(self, name, timezone_name):
        return {"id": "planned-id", "summary": name, "timeZone": timezone_name}


def test_calendar_sync_is_idempotent_updates_and_removes_events():
    root, database, config = _fixture()
    client = FakeCalendarClient()
    today = date(2026, 9, 24)

    first = sync_google_calendar(database, client=client, today=today, config_path=config)
    second = sync_google_calendar(database, client=client, today=today, config_path=config)

    assert (first.created, first.updated, first.unchanged) == (1, 0, 0)
    assert (second.created, second.updated, second.unchanged) == (0, 0, 1)
    assert client.created[0]["visibility"] == "private"
    assert client.created[0]["start"] == {"date": "2026-09-25"}
    assert client.created[0]["reminders"]["overrides"][0]["minutes"] == 720

    with connect(database) as connection:
        connection.execute("UPDATE workout_prescriptions SET title='Shortened ride' WHERE status='active'")
        connection.commit()
    changed = sync_google_calendar(database, client=client, today=today, config_path=config)
    assert changed.updated == 1
    assert client.updated[0][0] == "event-1"

    with connect(database) as connection:
        connection.execute("UPDATE workout_prescriptions SET prescribed_date='2026-10-30'")
        connection.commit()
    removed = sync_google_calendar(database, client=client, today=today, config_path=config)
    assert removed.removed == 1
    assert client.deleted == ["event-1"]


def test_calendar_dry_run_does_not_require_credentials_or_write_links():
    root, database, config = _fixture()
    result = sync_google_calendar(database, today=date(2026, 9, 24), config_path=config, dry_run=True)
    assert result.created == 1 and result.dry_run
    with connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM calendar_event_links").fetchone()[0] == 0


def test_calendar_status_and_windows_task_dry_run():
    root, database, config = _fixture()
    status = calendar_status(database, config)
    assert status["enabled"] is True
    assert status["authorized"] is False
    task = install_windows_tasks(30, dry_run=True)
    assert task.installed is False
    assert task.sync_minutes == 30
    assert len(task.commands) == 2


def test_separate_calendar_setup_and_activity_sync_are_idempotent():
    root, database, config = _fixture()
    setup = configure_google_calendars(client=FakeSetupClient(), config_path=config)
    assert setup["actual_calendar_id"] == "strava-id"
    assert setup["planned_calendar_id"] == "planned-id"
    assert setup["planned_calendar_created"] is True

    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        import_id = connection.execute(
            """INSERT INTO imports
               (source_filename, sha256, source_format, source, raw_path, imported_at, status, activity_count)
               VALUES ('ride.fit', 'activity-sha', 'fit', 'strava', 'ride.fit', ?, 'imported', 1)""",
            (now,),
        ).lastrowid
        connection.execute(
            """INSERT INTO activities
               (stable_id, source_activity_id, source_filename, raw_path, import_id,
                source_format, source, sport, name, started_at, elapsed_seconds,
                moving_seconds, distance_m, ascent_m, avg_hr, avg_power_w, imported_at)
               VALUES ('activity-1', '12345', 'ride.fit', 'ride.fit', ?, 'fit', 'strava',
                       'cycling', 'Morning Ride', '2026-09-12T08:00:00+00:00', 3900,
                       3600, 32000, 450, 151, 184, ?)""",
            (import_id, now),
        )
        connection.commit()

    client = FakeCalendarClient()
    first = sync_activity_calendar(
        database, date_from=date(2026, 9, 1), date_to=date(2026, 9, 30),
        client=client, config_path=config,
    )
    second = sync_activity_calendar(
        database, date_from=date(2026, 9, 1), date_to=date(2026, 9, 30),
        client=client, config_path=config,
    )
    assert (first.created, second.unchanged) == (1, 1)
    assert client.created[0]["summary"] == "🚴 Morning Ride"
    assert "32.00 km" in client.created[0]["description"]
    assert "1:05:00" in client.created[0]["description"]
    assert "strava.com/activities/12345" in client.created[0]["description"]
    assert client.created[0]["transparency"] == "transparent"


def _fixture():
    root = Path("data/test-runs") / uuid4().hex
    root.mkdir(parents=True)
    database = root / "calendar.sqlite3"
    config = root / "athlete.yaml"
    config.write_text(
        """google_calendar:
  enabled: true
  calendar_id: primary
  days_ahead: 14
  reminder_minutes: [720]
  client_file: training_data/auth/missing-client.json
  token_file: training_data/auth/missing-token.json
""",
        encoding="utf-8",
    )
    init_db(database)
    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        plan_id = connection.execute(
            """INSERT INTO training_plans
               (name, source_path, source_filename, source_sha256, imported_at, active)
               VALUES ('Test', 'test.xlsx', 'test.xlsx', 'abc', ?, 1)""",
            (now,),
        ).lastrowid
        session_id = connection.execute(
            """INSERT INTO planned_sessions
               (plan_id, planned_date, sport, title, source_sheet, source_row,
                source_key, raw_source_text, active, updated_at)
               VALUES (?, '2026-09-25', 'cycling', 'Threshold ride', 'Calendar', 1,
                       'test-1', 'Threshold ride', 1, ?)""",
            (plan_id, now),
        ).lastrowid
        connection.execute(
            """INSERT INTO workout_prescriptions
               (planned_session_id, prescribed_date, sport, title, description,
                duration_seconds, intensity, reason, status, action, confidence,
                evidence_json, rules_json, created_at)
               VALUES (?, '2026-09-25', 'cycling', 'Threshold ride', 'Three intervals',
                       3600, 'Threshold', 'Plan remains suitable', 'active', 'KEEP', 'high',
                       '[]', '[]', ?)""",
            (session_id, now),
        )
        connection.commit()
    return root, database, config
