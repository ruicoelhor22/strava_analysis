from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import webbrowser
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo

import requests
import yaml

from endurance_lab.config import load_athlete_config, paths, project_path
from endurance_lab.db import connect, init_db


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_API_ROOT = "https://www.googleapis.com/calendar/v3"
CALENDAR_SCOPES = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.calendars",
)
CALENDAR_SCOPE = " ".join(CALENDAR_SCOPES)
DEFAULT_CLIENT_FILE = "training_data/auth/google-calendar-client.json"
DEFAULT_TOKEN_FILE = "training_data/auth/google-calendar-token.json"


class GoogleCalendarError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class CalendarSyncSummary:
    enabled: bool = True
    calendar_id: str = "primary"
    date_from: str | None = None
    date_to: str | None = None
    considered: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    errors: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivityCalendarSyncSummary:
    enabled: bool = True
    calendar_id: str = ""
    date_from: str | None = None
    date_to: str | None = None
    considered: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    errors: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GoogleCalendarClient:
    def __init__(
        self,
        client_file: str | Path = DEFAULT_CLIENT_FILE,
        token_file: str | Path = DEFAULT_TOKEN_FILE,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.client_file = project_path(client_file)
        self.token_file = project_path(token_file)
        self.timeout = timeout
        self.session = session or requests.Session()

    def insert(self, calendar_id: str, event: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/calendars/{_quote(calendar_id)}/events", json=event)

    def update(self, calendar_id: str, event_id: str, event: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PUT", f"/calendars/{_quote(calendar_id)}/events/{_quote(event_id)}", json=event
        )

    def delete(self, calendar_id: str, event_id: str) -> None:
        self._request("DELETE", f"/calendars/{_quote(calendar_id)}/events/{_quote(event_id)}")

    def identity(self, calendar_id: str = "primary") -> dict[str, Any]:
        return self._request("GET", f"/calendars/{_quote(calendar_id)}")

    def list_calendars(self) -> list[dict[str, Any]]:
        calendars: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params = {"maxResults": 250}
            if page_token:
                params["pageToken"] = page_token
            payload = self._request("GET", "/users/me/calendarList", params=params)
            calendars.extend(payload.get("items") or [])
            page_token = payload.get("nextPageToken")
            if not page_token:
                return calendars

    def create_calendar(self, name: str, timezone_name: str) -> dict[str, Any]:
        return self._request(
            "POST", "/calendars", json={"summary": name, "timeZone": timezone_name}
        )

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        token = self._access_token()
        url = GOOGLE_API_ROOT + path
        response = None
        for attempt in range(3):
            response = self.session.request(
                method, url, headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout, **kwargs,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(2**attempt)
        assert response is not None
        if response.status_code == 401:
            raise GoogleCalendarError("GOOGLE_AUTH_REQUIRED", "Google Calendar authorization expired")
        if not 200 <= response.status_code < 300:
            detail = _response_error(response)
            raise GoogleCalendarError(
                "GOOGLE_CALENDAR_ERROR", f"Google Calendar returned HTTP {response.status_code}: {detail}"
            )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise GoogleCalendarError("GOOGLE_CALENDAR_ERROR", "Google Calendar returned invalid JSON") from exc

    def _access_token(self) -> str:
        token = _read_json(self.token_file, "Google Calendar token")
        expires_at = float(token.get("expires_at") or 0)
        if token.get("access_token") and expires_at > time.time() + 90:
            return str(token["access_token"])
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            raise GoogleCalendarError(
                "GOOGLE_AUTH_REQUIRED", "Run: python -m endurance_lab calendar-login"
            )
        client = _client_settings(self.client_file)
        response = self.session.post(
            client["token_uri"],
            data={
                "client_id": client["client_id"], "client_secret": client.get("client_secret", ""),
                "refresh_token": refresh_token, "grant_type": "refresh_token",
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise GoogleCalendarError(
                "GOOGLE_AUTH_REQUIRED", f"Google token refresh failed: {_response_error(response)}"
            )
        refreshed = response.json()
        token.update(refreshed)
        token["refresh_token"] = refreshed.get("refresh_token") or refresh_token
        token["expires_at"] = time.time() + float(refreshed.get("expires_in", 3600))
        _write_private_json(self.token_file, token)
        return str(token["access_token"])


def login_google_calendar(
    client_file: str | Path = DEFAULT_CLIENT_FILE,
    token_file: str | Path = DEFAULT_TOKEN_FILE,
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    client_path = project_path(client_file)
    token_path = project_path(token_file)
    client = _client_settings(client_path)
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    result: dict[str, str] = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            query = parse_qs(urlparse(self.path).query)
            result["state"] = query.get("state", [""])[0]
            result["code"] = query.get("code", [""])[0]
            result["error"] = query.get("error", [""])[0]
            body = (
                "<h2>Endurance Lab is connected to Google Calendar.</h2><p>You can close this window.</p>"
                if result["code"] else
                "<h2>Google Calendar authorization was not completed.</h2><p>You can close this window.</p>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Callback)
    server.timeout = timeout_seconds
    redirect_uri = f"http://127.0.0.1:{server.server_port}/"
    authorization_url = GOOGLE_AUTH_URL + "?" + urlencode({
        "client_id": client["client_id"], "redirect_uri": redirect_uri,
        "response_type": "code", "scope": CALENDAR_SCOPE,
        "access_type": "offline", "prompt": "consent", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })
    if not webbrowser.open(authorization_url):
        server.server_close()
        raise GoogleCalendarError("GOOGLE_AUTH_REQUIRED", f"Open this URL in a browser: {authorization_url}")
    server.handle_request()
    server.server_close()
    if result.get("state") != state:
        raise GoogleCalendarError("GOOGLE_AUTH_REQUIRED", "Google authorization state did not match")
    if not result.get("code"):
        raise GoogleCalendarError(
            "GOOGLE_AUTH_REQUIRED", result.get("error") or "Google authorization timed out"
        )
    response = requests.post(
        client["token_uri"],
        data={
            "client_id": client["client_id"], "client_secret": client.get("client_secret", ""),
            "code": result["code"], "code_verifier": verifier,
            "grant_type": "authorization_code", "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise GoogleCalendarError("GOOGLE_AUTH_REQUIRED", _response_error(response))
    token = response.json()
    token["expires_at"] = time.time() + float(token.get("expires_in", 3600))
    token["scope"] = token.get("scope") or CALENDAR_SCOPE
    _write_private_json(token_path, token)
    return {"authorized": True, "scope": token["scope"], "token_file": str(token_path)}


def calendar_enabled(config_path: str | Path | None = None) -> bool:
    return bool(load_athlete_config(config_path).get("google_calendar", {}).get("enabled", False))


def set_calendar_enabled(enabled: bool, config_path: str | Path | None = None) -> Path:
    target = project_path(config_path) if config_path else paths().athlete_config
    config = load_athlete_config(target)
    settings = config.setdefault("google_calendar", {})
    settings["enabled"] = bool(enabled)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def configure_google_calendars(
    actual_name: str = "strava",
    planned_name: str = "Endurance Lab - Planned",
    *,
    client: GoogleCalendarClient | Any | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the existing activity calendar and create the planned calendar if needed."""
    target = project_path(config_path) if config_path else paths().athlete_config
    config = load_athlete_config(target)
    settings = config.setdefault("google_calendar", {})
    api = client or GoogleCalendarClient(
        settings.get("client_file", DEFAULT_CLIENT_FILE),
        settings.get("token_file", DEFAULT_TOKEN_FILE),
    )
    calendars = api.list_calendars()
    actual = _calendar_named(calendars, actual_name)
    if not actual:
        available = ", ".join(sorted(str(item.get("summary")) for item in calendars if item.get("summary")))
        raise GoogleCalendarError(
            "GOOGLE_CALENDAR_NOT_FOUND",
            f"Calendar '{actual_name}' was not found. Available calendars: {available or 'none'}",
        )
    planned = _calendar_named(calendars, planned_name)
    created = False
    if not planned:
        timezone_name = str(config.get("athlete", {}).get("timezone") or "Europe/Lisbon")
        planned = api.create_calendar(planned_name, timezone_name)
        created = True
    actual_id = str(actual.get("id") or "")
    planned_id = str(planned.get("id") or "")
    if not actual_id or not planned_id:
        raise GoogleCalendarError("GOOGLE_CALENDAR_ERROR", "Google returned a calendar without an ID")
    if actual_id == planned_id:
        raise GoogleCalendarError(
            "GOOGLE_CALENDAR_ERROR", "Completed and planned sessions must use different calendars"
        )
    settings.update({
        "enabled": True,
        "actual_calendar_name": str(actual.get("summary") or actual_name),
        "actual_calendar_id": actual_id,
        "planned_calendar_name": str(planned.get("summary") or planned_name),
        "planned_calendar_id": planned_id,
        # Keep the old key for compatibility with older app versions.
        "calendar_id": planned_id,
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(target)
    return {
        "enabled": True,
        "actual_calendar": settings["actual_calendar_name"],
        "actual_calendar_id": actual_id,
        "planned_calendar": settings["planned_calendar_name"],
        "planned_calendar_id": planned_id,
        "planned_calendar_created": created,
        "config": str(target),
    }


def sync_google_calendar(
    database: str | Path | None = None,
    *,
    client: GoogleCalendarClient | Any | None = None,
    today: date | None = None,
    days_ahead: int | None = None,
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> CalendarSyncSummary:
    init_db(database)
    config = load_athlete_config(config_path)
    settings = config.get("google_calendar", {})
    planned_id = settings.get("planned_calendar_id")
    calendar_id = str(planned_id or settings.get("calendar_id") or "")
    if not calendar_id and not dry_run:
        raise GoogleCalendarError(
            "GOOGLE_CALENDAR_NOT_CONFIGURED",
            "Planned-session calendar is not configured. Run: python -m endurance_lab calendar-setup",
        )
    calendar_id = calendar_id or "not configured"
    today = today or date.today()
    horizon = max(0, int(days_ahead if days_ahead is not None else settings.get("days_ahead", 14)))
    end = today + timedelta(days=horizon)
    summary = CalendarSyncSummary(
        enabled=bool(settings.get("enabled", False)), calendar_id=calendar_id,
        date_from=today.isoformat(), date_to=end.isoformat(), dry_run=dry_run,
    )
    rows = _active_prescriptions(database, today, end)
    summary.considered = len(rows)
    api = client
    if not dry_run and api is None:
        api = GoogleCalendarClient(
            settings.get("client_file", DEFAULT_CLIENT_FILE),
            settings.get("token_file", DEFAULT_TOKEN_FILE),
        )
    desired_ids: set[int] = set()
    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        for row in rows:
            planned_id = int(row["planned_session_id"])
            desired_ids.add(planned_id)
            event = _event(row, settings)
            digest = hashlib.sha256(
                json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            link = connection.execute(
                """SELECT * FROM calendar_event_links
                   WHERE provider='google' AND calendar_id=? AND planned_session_id=?""",
                (calendar_id, planned_id),
            ).fetchone()
            if link and link["status"] == "active" and link["content_hash"] == digest:
                summary.unchanged += 1
                continue
            if dry_run:
                summary.updated += int(link is not None)
                summary.created += int(link is None)
                continue
            try:
                if link:
                    remote = api.update(calendar_id, str(link["external_event_id"]), event)
                    event_id = str(remote.get("id") or link["external_event_id"])
                    summary.updated += 1
                else:
                    remote = api.insert(calendar_id, event)
                    event_id = str(remote.get("id") or "")
                    if not event_id:
                        raise GoogleCalendarError("GOOGLE_CALENDAR_ERROR", "Created event has no ID")
                    summary.created += 1
                connection.execute(
                    """INSERT INTO calendar_event_links
                       (provider, calendar_id, planned_session_id, prescription_id, external_event_id,
                        content_hash, event_date, status, last_synced_at, error)
                       VALUES ('google', ?, ?, ?, ?, ?, ?, 'active', ?, NULL)
                       ON CONFLICT(provider, calendar_id, planned_session_id) DO UPDATE SET
                         prescription_id=excluded.prescription_id, external_event_id=excluded.external_event_id,
                         content_hash=excluded.content_hash, event_date=excluded.event_date,
                         status='active', last_synced_at=excluded.last_synced_at, error=NULL""",
                    (calendar_id, planned_id, row["prescription_id"], event_id, digest,
                     row["prescribed_date"], now),
                )
                connection.commit()
            except GoogleCalendarError as exc:
                summary.errors += 1
                if exc.code == "GOOGLE_AUTH_REQUIRED":
                    raise

        stale = connection.execute(
            """SELECT * FROM calendar_event_links
               WHERE provider='google' AND calendar_id=? AND status='active'
                 AND event_date BETWEEN ? AND ?""",
            (calendar_id, today.isoformat(), end.isoformat()),
        ).fetchall()
        for link in stale:
            if int(link["planned_session_id"]) in desired_ids:
                continue
            if not dry_run:
                try:
                    api.delete(calendar_id, str(link["external_event_id"]))
                    connection.execute(
                        "UPDATE calendar_event_links SET status='removed', last_synced_at=?, error=NULL WHERE id=?",
                        (now, link["id"]),
                    )
                    connection.commit()
                except GoogleCalendarError as exc:
                    summary.errors += 1
                    if exc.code == "GOOGLE_AUTH_REQUIRED":
                        raise
                    continue
            summary.removed += 1
    return summary


def sync_activity_calendar(
    database: str | Path | None = None,
    *,
    date_from: date,
    date_to: date,
    client: GoogleCalendarClient | Any | None = None,
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> ActivityCalendarSyncSummary:
    """Create or update completed activity events for an inclusive local-date range."""
    if date_to < date_from:
        raise ValueError("Calendar activity end date must not be before the start date")
    init_db(database)
    config = load_athlete_config(config_path)
    settings = config.get("google_calendar", {})
    calendar_id = str(settings.get("actual_calendar_id") or "")
    if not calendar_id and not dry_run:
        raise GoogleCalendarError(
            "GOOGLE_CALENDAR_NOT_CONFIGURED",
            "Completed-activity calendar is not configured. Run: python -m endurance_lab calendar-setup",
        )
    summary = ActivityCalendarSyncSummary(
        enabled=bool(settings.get("enabled", False)), calendar_id=calendar_id or "not configured",
        date_from=date_from.isoformat(), date_to=date_to.isoformat(), dry_run=dry_run,
    )
    rows = _completed_activities(database, date_from, date_to, config)
    summary.considered = len(rows)
    api = client
    if not dry_run and api is None:
        api = GoogleCalendarClient(
            settings.get("client_file", DEFAULT_CLIENT_FILE),
            settings.get("token_file", DEFAULT_TOKEN_FILE),
        )
    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        for row in rows:
            activity_id = int(row["id"])
            event = _activity_event(row, config)
            digest = hashlib.sha256(
                json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            link = connection.execute(
                """SELECT * FROM activity_calendar_event_links
                   WHERE provider='google' AND calendar_id=? AND activity_id=?""",
                (calendar_id, activity_id),
            ).fetchone()
            if link and link["status"] == "active" and link["content_hash"] == digest:
                summary.unchanged += 1
                continue
            if dry_run:
                summary.updated += int(link is not None)
                summary.created += int(link is None)
                continue
            try:
                if link:
                    remote = api.update(calendar_id, str(link["external_event_id"]), event)
                    event_id = str(remote.get("id") or link["external_event_id"])
                    summary.updated += 1
                else:
                    remote = api.insert(calendar_id, event)
                    event_id = str(remote.get("id") or "")
                    if not event_id:
                        raise GoogleCalendarError("GOOGLE_CALENDAR_ERROR", "Created event has no ID")
                    summary.created += 1
                connection.execute(
                    """INSERT INTO activity_calendar_event_links
                       (provider, calendar_id, activity_id, external_event_id, content_hash,
                        event_date, status, last_synced_at, error)
                       VALUES ('google', ?, ?, ?, ?, ?, 'active', ?, NULL)
                       ON CONFLICT(provider, calendar_id, activity_id) DO UPDATE SET
                         external_event_id=excluded.external_event_id,
                         content_hash=excluded.content_hash, event_date=excluded.event_date,
                         status='active', last_synced_at=excluded.last_synced_at, error=NULL""",
                    (calendar_id, activity_id, event_id, digest, event["start"]["dateTime"][:10], now),
                )
                connection.commit()
            except GoogleCalendarError as exc:
                summary.errors += 1
                if exc.code == "GOOGLE_AUTH_REQUIRED":
                    raise
    return summary


def calendar_status(database: str | Path | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    init_db(database)
    settings = load_athlete_config(config_path).get("google_calendar", {})
    token_path = project_path(settings.get("token_file", DEFAULT_TOKEN_FILE))
    client_path = project_path(settings.get("client_file", DEFAULT_CLIENT_FILE))
    with connect(database) as connection:
        counts = connection.execute(
            """SELECT COUNT(*) total, SUM(status='active') active, MAX(last_synced_at) last_synced
               FROM calendar_event_links WHERE provider='google'"""
        ).fetchone()
        activity_counts = connection.execute(
            """SELECT COUNT(*) total, SUM(status='active') active, MAX(last_synced_at) last_synced
               FROM activity_calendar_event_links WHERE provider='google'"""
        ).fetchone()
    return {
        "enabled": bool(settings.get("enabled", False)),
        "calendar_id": str(settings.get("planned_calendar_id") or settings.get("calendar_id") or "not configured"),
        "planned_calendar_name": str(settings.get("planned_calendar_name") or "Planned sessions"),
        "actual_calendar_id": str(settings.get("actual_calendar_id") or ""),
        "actual_calendar_name": str(settings.get("actual_calendar_name") or "strava"),
        "client_configured": client_path.exists(), "authorized": token_path.exists(),
        "linked_events": int(counts["total"] or 0), "active_events": int(counts["active"] or 0),
        "last_synced": counts["last_synced"],
        "activity_linked_events": int(activity_counts["total"] or 0),
        "active_activity_events": int(activity_counts["active"] or 0),
        "activities_last_synced": activity_counts["last_synced"],
    }


def _active_prescriptions(database, start: date, end: date) -> list[dict[str, Any]]:
    with connect(database) as connection:
        return [dict(row) for row in connection.execute(
            """SELECT p.id prescription_id, p.planned_session_id, p.prescribed_date,
                      p.sport, p.title, p.description, p.duration_seconds, p.distance_m,
                      p.intensity, p.targets_json, p.reason, p.action, p.confidence,
                      p.optional_gate_json, s.priority
               FROM workout_prescriptions p
               JOIN planned_sessions s ON s.id=p.planned_session_id
               WHERE p.status='active' AND s.active=1
                 AND p.prescribed_date BETWEEN ? AND ?
               ORDER BY p.prescribed_date, p.id""",
            (start.isoformat(), end.isoformat()),
        )]


def _event(row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    event_day = date.fromisoformat(str(row["prescribed_date"]))
    duration = float(row.get("duration_seconds") or 0)
    lines = [
        f"Sport: {str(row.get('sport') or 'other').title()}",
        f"Decision: {row.get('action') or 'KEEP'}",
    ]
    if duration:
        lines.append(f"Duration: {round(duration / 60)} minutes")
    if row.get("intensity"):
        lines.append(f"Intensity: {row['intensity']}")
    if row.get("description"):
        lines.extend(["", str(row["description"])])
    if row.get("reason"):
        lines.extend(["", "Coach: " + str(row["reason"])])
    event: dict[str, Any] = {
        "summary": f"{str(row.get('sport') or 'Training').title()}: {row['title']}",
        "description": "\n".join(lines),
        "start": {"date": event_day.isoformat()},
        "end": {"date": (event_day + timedelta(days=1)).isoformat()},
        "visibility": "private",
        "extendedProperties": {"private": {
            "enduranceLab": "true", "plannedSessionId": str(row["planned_session_id"]),
            "prescriptionId": str(row["prescription_id"]),
        }},
    }
    reminders = [int(value) for value in settings.get("reminder_minutes", [720])]
    event["reminders"] = {
        "useDefault": False,
        "overrides": [{"method": "popup", "minutes": value} for value in reminders],
    }
    color_id = settings.get("color_id")
    if color_id:
        event["colorId"] = str(color_id)
    return event


def _calendar_named(calendars: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    wanted = name.strip().casefold()
    return next(
        (item for item in calendars if str(item.get("summary") or "").strip().casefold() == wanted),
        None,
    )


def _completed_activities(
    database: str | Path | None,
    date_from: date,
    date_to: date,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    zone = ZoneInfo(str(config.get("athlete", {}).get("timezone") or "UTC"))
    utc_start = datetime.combine(date_from, datetime.min.time(), zone).astimezone(timezone.utc)
    utc_end = datetime.combine(date_to + timedelta(days=1), datetime.min.time(), zone).astimezone(timezone.utc)
    with connect(database) as connection:
        return [dict(row) for row in connection.execute(
            """SELECT id, source_activity_id, name, sport, started_at, ended_at,
                      elapsed_seconds, moving_seconds, distance_m, ascent_m, calories,
                      avg_hr, max_hr, avg_speed_mps, avg_power_w, source, metadata_only
               FROM activities
               WHERE started_at >= ? AND started_at < ?
               ORDER BY started_at, id""",
            (utc_start.isoformat(), utc_end.isoformat()),
        )]


def _activity_event(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    timezone_name = str(config.get("athlete", {}).get("timezone") or "UTC")
    zone = ZoneInfo(timezone_name)
    start = datetime.fromisoformat(str(row["started_at"]).replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    start = start.astimezone(zone)
    elapsed = float(row.get("elapsed_seconds") or 0)
    moving = float(row.get("moving_seconds") or 0)
    if elapsed > 0:
        end = start + timedelta(seconds=max(60, elapsed))
    elif row.get("ended_at"):
        end = datetime.fromisoformat(str(row["ended_at"]).replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        end = end.astimezone(zone)
        if end <= start:
            end = start + timedelta(seconds=max(60, moving or 3600))
    else:
        end = start + timedelta(seconds=max(60, moving or 3600))
    duration = elapsed or moving
    distance = float(row.get("distance_m") or 0)
    sport = str(row.get("sport") or "other").lower()
    icon, sport_label = _sport_display(sport)
    title = str(row.get("name") or "").strip()
    if not title or title == str(row.get("source_activity_id") or ""):
        title = sport_label
    lines = [f"{icon} {sport_label}"]
    if duration:
        lines.append(f"⏱ Total time: {_duration_text(duration)}")
    if moving and duration and abs(duration - moving) >= 60:
        lines.append(f"▶ Moving time: {_duration_text(moving)}")
    if distance:
        lines.append(f"📏 Distance: {distance / 1000:.2f} km")
    if row.get("ascent_m") is not None:
        lines.append(f"↗ Elevation gain: {float(row['ascent_m']):.0f} m")
    speed = float(row.get("avg_speed_mps") or 0)
    if speed and sport == "running":
        pace_minutes, pace_remainder = divmod(round(1000 / speed), 60)
        lines.append(f"🏃 Average pace: {pace_minutes}:{pace_remainder:02d} /km")
    elif speed:
        lines.append(f"💨 Average speed: {speed * 3.6:.1f} km/h")
    if row.get("avg_hr") is not None:
        heart_rate = f"{float(row['avg_hr']):.0f} bpm average"
        if row.get("max_hr") is not None:
            heart_rate += f" · {float(row['max_hr']):.0f} bpm max"
        lines.append(f"❤️ Heart rate: {heart_rate}")
    if row.get("avg_power_w") is not None:
        lines.append(f"⚡ Average power: {float(row['avg_power_w']):.0f} W")
    if row.get("calories") is not None:
        lines.append(f"🔥 Energy: {float(row['calories']):.0f} kcal")
    if row.get("source_activity_id"):
        lines.extend(["", "View activity on Strava:", f"https://www.strava.com/activities/{row['source_activity_id']}"])
    return {
        "summary": f"{icon} {title}",
        "description": "\n".join(lines),
        "start": {"dateTime": start.isoformat(), "timeZone": timezone_name},
        "end": {"dateTime": end.isoformat(), "timeZone": timezone_name},
        "visibility": "private",
        "transparency": "transparent",
        "reminders": {"useDefault": False, "overrides": []},
        "extendedProperties": {"private": {
            "enduranceLab": "true", "activityId": str(row["id"]),
            "stravaActivityId": str(row.get("source_activity_id") or ""),
        }},
    }


def _duration_text(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _sport_display(sport: str) -> tuple[str, str]:
    return {
        "cycling": ("🚴", "Cycling"),
        "running": ("🏃", "Running"),
        "strength": ("🏋️", "Strength training"),
        "swimming": ("🏊", "Swimming"),
        "hiking": ("🥾", "Hiking"),
        "walking": ("🚶", "Walking"),
    }.get(sport, ("🏅", "Training"))


def _client_settings(path: Path) -> dict[str, str]:
    value = _read_json(path, "Google OAuth client")
    settings = value.get("installed") or value.get("web") or value
    client_id = str(settings.get("client_id") or "")
    if not client_id:
        raise GoogleCalendarError(
            "GOOGLE_CLIENT_MISSING",
            f"Download a Google OAuth desktop client JSON to: {path}",
        )
    return {
        "client_id": client_id,
        "client_secret": str(settings.get("client_secret") or ""),
        "token_uri": str(settings.get("token_uri") or GOOGLE_TOKEN_URL),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise GoogleCalendarError("GOOGLE_AUTH_REQUIRED", f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoogleCalendarError("GOOGLE_AUTH_REQUIRED", f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GoogleCalendarError("GOOGLE_AUTH_REQUIRED", f"Invalid {label}: {path}")
    return value


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _response_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("error_description") or payload.get("error", {}).get("message") or payload.get("error"))[:500]
    except (ValueError, AttributeError):
        return response.text[:500]


def _quote(value: str) -> str:
    from urllib.parse import quote
    return quote(value, safe="")
