from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from endurance_lab.db import connect, init_db
from endurance_lab.session_classifier import QUALITY_CLASSES, classify_activity


def analyze_intervals(
    activity_id: int,
    database: str | Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    init_db(database)
    classification = classify_activity(activity_id, database, persist=persist)
    with connect(database) as connection:
        context = connection.execute(
            """SELECT a.sport, p.id AS planned_session_id, p.sport AS planned_sport,
                      p.title, p.description, p.interval_structure, p.intensity
               FROM activities a
               LEFT JOIN planned_activity_matches m ON m.activity_id = a.id AND m.is_selected = 1
               LEFT JOIN planned_sessions p ON p.id = m.planned_session_id
               WHERE a.id = ?""",
            (activity_id,),
        ).fetchone()
        if not context:
            raise ValueError(f"Activity {activity_id} was not found")
        if context["planned_session_id"] and context["planned_sport"] != context["sport"]:
            return _empty(activity_id, "actual sport differs from the linked planned session")
        if classification.classification not in QUALITY_CLASSES:
            return _empty(activity_id, "activity is not classified as a quality session")
        planned_text = " ".join(
            str(context[key] or "") for key in ("title", "description", "interval_structure", "intensity")
        ).lower()
        if context["planned_session_id"] and not any(
            token in planned_text for token in ("threshold", "treshold", "vo2", "interval", "tempo", "race")
        ):
            return _empty(activity_id, "linked plan does not describe endurance quality work")
        laps = [dict(row) for row in connection.execute(
            """SELECT * FROM activity_laps WHERE activity_id = ?
               AND duration_seconds >= 30 ORDER BY lap_number""",
            (activity_id,),
        )]
        targets = [dict(row) for row in connection.execute(
            """SELECT t.* FROM planned_session_targets t
               JOIN planned_activity_matches m ON m.planned_session_id = t.planned_session_id
               WHERE m.activity_id = ? AND m.is_selected = 1""",
            (activity_id,),
        )]
        if len(laps) < 2:
            return _empty(activity_id, "fewer than two usable laps")
        expected = _expected_count(context)
        work = _select_work_laps(laps, targets, str(context["sport"]), expected)
        if not work:
            return _empty(activity_id, "laps do not contain a defensible work-interval signal")
        intervals = [_interval(index, lap, targets, str(context["sport"])) for index, lap in enumerate(work, 1)]
        values = [_output_value(item, str(context["sport"])) for item in intervals]
        values = [value for value in values if value is not None]
        fade = None
        cv = None
        if len(values) >= 2 and values[0]:
            fade = (values[-1] - values[0]) / values[0] * 100
            cv = statistics.pstdev(values) / statistics.mean(values) * 100 if statistics.mean(values) else None
        result = {
            "activity_id": activity_id,
            "planned_session_id": context["planned_session_id"],
            "expected_intervals": expected,
            "detected_intervals": len(intervals),
            "intervals": intervals,
            "output_change_pct": fade,
            "output_cv_pct": cv,
            "confidence": "high" if expected and len(intervals) == expected else "medium",
            "reason": "work laps selected using planned targets" if targets else "work laps selected from repeated high-output laps",
        }
        if persist:
            connection.execute("DELETE FROM activity_intervals WHERE activity_id = ?", (activity_id,))
            for item in intervals:
                connection.execute(
                    """INSERT INTO activity_intervals (
                           activity_id, interval_number, lap_number, duration_seconds,
                           distance_m, avg_power_w, avg_hr, avg_pace_seconds_per_km,
                           avg_cadence, target_adherence, evidence_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        activity_id, item["interval_number"], item["lap_number"], item["duration_seconds"],
                        item["distance_m"], item["avg_power_w"], item["avg_hr"],
                        item["avg_pace_seconds_per_km"], item["avg_cadence"],
                        item["target_adherence"], json.dumps(item["evidence"], ensure_ascii=False),
                    ),
                )
            connection.commit()
        return result


def _select_work_laps(laps, targets, sport: str, expected: int | None):
    power = next((target for target in targets if target["target_type"] == "power" and target["minimum_value"]), None)
    candidates = []
    if power:
        floor = float(power["minimum_value"]) * 0.80
        candidates = [lap for lap in laps if lap.get("avg_power_w") and float(lap["avg_power_w"]) >= floor]
    elif sport == "running":
        pace_laps = [lap for lap in laps if lap.get("distance_m") and lap.get("duration_seconds")]
        paces = [_pace(lap) for lap in pace_laps]
        if len(paces) >= 3:
            median = statistics.median(paces)
            candidates = [lap for lap in pace_laps if _pace(lap) <= median * 0.92]
    else:
        powered = [lap for lap in laps if lap.get("avg_power_w") and lap.get("duration_seconds", 0) >= 60]
        if len(powered) >= 3:
            median = statistics.median(float(lap["avg_power_w"]) for lap in powered)
            candidates = [lap for lap in powered if float(lap["avg_power_w"]) >= median * 1.08]
    if expected and len(candidates) > expected:
        candidates = sorted(candidates, key=lambda lap: int(lap["lap_number"]))[:expected]
    return sorted(candidates, key=lambda lap: int(lap["lap_number"]))


def _interval(number: int, lap: dict[str, Any], targets, sport: str) -> dict[str, Any]:
    pace = _pace(lap) if lap.get("distance_m") and lap.get("duration_seconds") else None
    adherence = "unknown"
    evidence = []
    power = next((target for target in targets if target["target_type"] == "power"), None)
    if power and lap.get("avg_power_w") and power.get("minimum_value") is not None and power.get("maximum_value") is not None:
        actual = float(lap["avg_power_w"])
        minimum, maximum = float(power["minimum_value"]), float(power["maximum_value"])
        adherence = "below" if actual < minimum * 0.95 else "above" if actual > maximum * 1.05 else "close"
        evidence.append(f"lap power {actual:.0f} W versus {minimum:.0f}-{maximum:.0f} W target")
    return {
        "interval_number": number,
        "lap_number": int(lap["lap_number"]),
        "duration_seconds": lap.get("duration_seconds"),
        "distance_m": lap.get("distance_m"),
        "avg_power_w": lap.get("avg_power_w"),
        "avg_hr": lap.get("avg_hr"),
        "avg_pace_seconds_per_km": pace,
        "avg_cadence": lap.get("avg_cadence"),
        "target_adherence": adherence,
        "evidence": evidence,
    }


def _expected_count(context) -> int | None:
    text = " ".join(str(context[key] or "") for key in ("title", "description", "interval_structure"))
    match = re.search(r"\b(\d{1,2})\s*[x×]\s*\d", text.lower())
    return int(match.group(1)) if match else None


def _pace(lap) -> float:
    return float(lap["duration_seconds"]) / (float(lap["distance_m"]) / 1000)


def _output_value(interval, sport: str) -> float | None:
    if sport == "running" and interval["avg_pace_seconds_per_km"]:
        return 1000 / float(interval["avg_pace_seconds_per_km"])
    return float(interval["avg_power_w"]) if interval["avg_power_w"] is not None else None


def _empty(activity_id: int, reason: str) -> dict[str, Any]:
    return {
        "activity_id": activity_id, "planned_session_id": None,
        "expected_intervals": None, "detected_intervals": 0, "intervals": [],
        "output_change_pct": None, "output_cv_pct": None,
        "confidence": "low", "reason": reason,
    }
