from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from endurance_lab.db import connect, init_db


def match_training_plan(database: str | Path | None = None) -> dict[str, int]:
    init_db(database)
    now = datetime.now(timezone.utc).isoformat()
    counts: Counter[str] = Counter()
    with connect(database) as connection:
        sessions = connection.execute(
            "SELECT * FROM planned_sessions WHERE active = 1 ORDER BY planned_date, id"
        ).fetchall()
        manually_matched = {
            int(row[0]) for row in connection.execute(
                "SELECT planned_session_id FROM planned_activity_matches WHERE match_status = 'manual' AND is_selected = 1"
            )
        }
        assigned = {
            int(row[0]) for row in connection.execute(
                "SELECT activity_id FROM planned_activity_matches WHERE match_status = 'manual' AND is_selected = 1"
            )
        }

        exact: list[tuple[Any, Any]] = []
        remaining = []
        for session in sessions:
            session_id = int(session["id"])
            if session_id in manually_matched:
                counts["manual"] += 1
                continue
            actual = _json(session["workbook_actual_json"])
            source_id = str(actual.get("Strava Activity ID") or "").strip()
            activity = connection.execute(
                "SELECT * FROM activities WHERE source_activity_id = ?", (source_id,)
            ).fetchone() if source_id else None
            if activity:
                exact.append((session, activity))
                assigned.add(int(activity["id"]))
            else:
                remaining.append(session)

        for session, activity in exact:
            _replace_automatic_matches(connection, int(session["id"]))
            reason = {
                "signals": ["exact Strava activity ID supplied by coaching workbook"],
                "source_activity_id": activity["source_activity_id"],
            }
            _insert_match(connection, session, activity, 100.0, "workbook_strava_id", "matched", reason, True, now)
            counts["matched"] += 1

        for session in remaining:
            session_id = int(session["id"])
            _replace_automatic_matches(connection, session_id)
            candidates = _candidates(connection, session, assigned)
            scored = sorted(
                ((_score(session, candidate), candidate) for candidate in candidates),
                key=lambda item: item[0][0], reverse=True,
            )
            if not scored or scored[0][0][0] < 55:
                counts["unmatched"] += 1
                continue
            (top_score, top_reason), top = scored[0]
            second_score = scored[1][0][0] if len(scored) > 1 else None
            ambiguous = second_score is not None and top_score - second_score <= 5
            if ambiguous:
                for (score, reason), candidate in scored[:2]:
                    _insert_match(
                        connection, session, candidate, score, "deterministic_score",
                        "ambiguous", reason, False, now,
                    )
                counts["ambiguous"] += 1
            else:
                status = "matched" if top_score >= 75 else "probable"
                _insert_match(
                    connection, session, top, top_score, "deterministic_score",
                    status, top_reason, True, now,
                )
                assigned.add(int(top["id"]))
                counts[status] += 1
        connection.commit()
    for key in ("matched", "probable", "ambiguous", "unmatched", "manual"):
        counts.setdefault(key, 0)
    return dict(counts)


def set_manual_match(
    planned_session_id: int,
    activity_id: int,
    notes: str | None = None,
    database: str | Path | None = None,
) -> None:
    init_db(database)
    now = datetime.now(timezone.utc).isoformat()
    with connect(database) as connection:
        connection.execute(
            "UPDATE planned_activity_matches SET is_selected = 0 WHERE planned_session_id = ?",
            (planned_session_id,),
        )
        connection.execute(
            """INSERT INTO planned_activity_matches (
                   planned_session_id, activity_id, match_score, match_method, match_status,
                   reason_json, is_selected, matched_at, manual_notes
               ) VALUES (?, ?, 100, 'manual_override', 'manual', ?, 1, ?, ?)
               ON CONFLICT(planned_session_id, activity_id) DO UPDATE SET
                   match_score = 100, match_method = 'manual_override', match_status = 'manual',
                   reason_json = excluded.reason_json, is_selected = 1,
                   matched_at = excluded.matched_at, manual_notes = excluded.manual_notes""",
            (planned_session_id, activity_id,
             json.dumps({"signals": ["manual override"]}), now, notes),
        )
        connection.commit()


def _candidates(connection, session, assigned: set[int]):
    planned = date.fromisoformat(str(session["planned_date"]))
    start = (planned - timedelta(days=1)).isoformat()
    end = (planned + timedelta(days=2)).isoformat()
    rows = connection.execute(
        """SELECT a.*, d.normalized_power_w, d.selected_load
           FROM activities a
           LEFT JOIN derived_activity_metrics d ON d.activity_id = a.id
           WHERE substr(a.started_at, 1, 10) >= ? AND substr(a.started_at, 1, 10) < ?
           ORDER BY a.started_at""",
        (start, end),
    ).fetchall()
    planned_sport = str(session["sport"])
    return [
        row for row in rows
        if int(row["id"]) not in assigned
        and (planned_sport == "other" or str(row["sport"]) == planned_sport)
    ]


def _score(session, activity) -> tuple[float, dict[str, Any]]:
    score = 0.0
    signals: list[str] = []
    planned_date = date.fromisoformat(str(session["planned_date"]))
    actual_date = datetime.fromisoformat(str(activity["started_at"])).date()
    delta = abs((actual_date - planned_date).days)
    if delta == 0:
        score += 45
        signals.append("same calendar date (+45)")
    elif delta == 1:
        score += 15
        signals.append("adjacent calendar date (+15)")

    if str(session["sport"]) == str(activity["sport"]):
        score += 30
        signals.append("same normalized sport (+30)")
    elif str(session["sport"]) == "other":
        score += 4
        signals.append("flexible recovery/other sport (+4)")

    planned_duration = _float(session["planned_duration_seconds"])
    actual_duration = _float(activity["moving_seconds"]) or _float(activity["elapsed_seconds"])
    if planned_duration and actual_duration:
        difference = abs(actual_duration - planned_duration) / planned_duration
        points = 15 if difference <= 0.10 else 10 if difference <= 0.25 else 4 if difference <= 0.50 else 0
        score += points
        if points:
            signals.append(f"duration within {difference:.0%} (+{points})")

    planned_distance = _float(session["planned_distance_m"])
    actual_distance = _float(activity["distance_m"])
    if planned_distance and actual_distance:
        difference = abs(actual_distance - planned_distance) / planned_distance
        points = 8 if difference <= 0.15 else 4 if difference <= 0.35 else 0
        score += points
        if points:
            signals.append(f"distance within {difference:.0%} (+{points})")

    planned_tokens = _tokens(" ".join(str(session[key] or "") for key in ("title", "description", "session_type")))
    actual_tokens = _tokens(str(activity["name"] or ""))
    overlap = planned_tokens & actual_tokens
    if overlap:
        points = min(7, len(overlap) * 2)
        score += points
        signals.append(f"title keywords {sorted(overlap)} (+{points})")
    return min(score, 100.0), {
        "signals": signals,
        "planned_date": planned_date.isoformat(),
        "actual_date": actual_date.isoformat(),
        "planned_sport": session["sport"],
        "actual_sport": activity["sport"],
    }


def _replace_automatic_matches(connection, session_id: int) -> None:
    connection.execute(
        "DELETE FROM planned_activity_matches WHERE planned_session_id = ? AND match_status != 'manual'",
        (session_id,),
    )


def _insert_match(connection, session, activity, score, method, status, reason, selected, now):
    connection.execute(
        """INSERT INTO planned_activity_matches (
               planned_session_id, activity_id, match_score, match_method, match_status,
               reason_json, is_selected, matched_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(planned_session_id, activity_id) DO UPDATE SET
               match_score = excluded.match_score, match_method = excluded.match_method,
               match_status = excluded.match_status, reason_json = excluded.reason_json,
               is_selected = excluded.is_selected, matched_at = excluded.matched_at""",
        (int(session["id"]), int(activity["id"]), score, method, status,
         json.dumps(reason, ensure_ascii=False), int(selected), now),
    )


def _tokens(value: str) -> set[str]:
    stop = {"the", "and", "with", "session", "workout", "easy", "min", "ride", "run", "gym"}
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) >= 3 and token not in stop}


def _float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _json(value) -> dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
        return loaded if isinstance(loaded, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}
