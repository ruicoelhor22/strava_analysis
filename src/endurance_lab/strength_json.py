from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from endurance_lab.identity import source_activity_id_from_filename
from endurance_lab.models import ParsedActivity, StrengthSet
from endurance_lab.normalize import finalize_activity


def parse_strength_json(
    path: str | Path, file_sha256: str | None = None
) -> list[ParsedActivity]:
    source = Path(path)
    sha256 = file_sha256 or hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {source.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Strength JSON must contain an object: {source.name}")

    started_at = _datetime(_first_present(payload, "start_time", "started_at", "start_date"))
    if started_at is None:
        raise ValueError(f"Strength JSON has no valid start timestamp: {source.name}")
    elapsed = _number(_first_present(payload, "elapsed_time", "duration"))
    active = _number(payload.get("active_time"))
    source_id = source_activity_id_from_filename(source.name)
    creator = payload.get("creator") if isinstance(payload.get("creator"), dict) else {}
    provider = str(creator.get("name") or payload.get("provider") or "json")
    sets = _parse_sets(payload.get("sets"), started_at)
    activity = ParsedActivity(
        stable_id=f"strava:{source_id}" if source_id else f"json:{sha256}:0",
        source_activity_id=source_id,
        sport="strength",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=elapsed) if elapsed is not None else None,
        name=_name(payload, source),
        elapsed_seconds=elapsed,
        moving_seconds=active or elapsed,
        calories=_number(_first_present(payload, "total_calories", "calories")),
        avg_hr=_number(_first_present(payload, "average_heart_rate", "avg_hr")),
        max_hr=_number(_first_present(payload, "maximum_heart_rate", "max_hr")),
        device=provider,
        source_metadata={
            "provider": provider,
            "version": payload.get("version"),
            "utc_offset_seconds": payload.get("utc_offset"),
            "schema": "hevy_strava_original" if provider.lower() == "hevy" else "generic_strength_json",
        },
        strength_sets=sets,
    )
    return [finalize_activity(activity)]


def _parse_sets(value: Any, activity_start: datetime) -> list[StrengthSet]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Strength JSON field 'sets' must be a list")
    counts: defaultdict[str, int] = defaultdict(int)
    parsed: list[StrengthSet] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        exercise = raw.get("exercise_type") or raw.get("exercise") or raw.get("name")
        if not exercise:
            continue
        exercise_name = str(exercise).strip()
        counts[exercise_name] += 1
        parsed.append(
            StrengthSet(
                exercise_name=exercise_name,
                set_number=counts[exercise_name],
                repetitions=_integer(_first_present(raw, "repetitions", "reps")),
                load_value=_number(raw.get("weight") if "weight" in raw else raw.get("load")),
                # The observed Hevy export has no unit field. Preserve the number without guessing kg/lb.
                load_unit=_text(raw.get("weight_unit") or raw.get("unit")),
                started_at=_datetime(raw.get("start_time")) or activity_start,
                duration_seconds=_number(_first_present(raw, "duration", "duration_seconds")),
                metadata={
                    key: raw[key]
                    for key in ("set_type", "rpe", "distance", "duration")
                    if raw.get(key) is not None
                },
            )
        )
    return parsed


def _name(payload: dict[str, Any], source: Path) -> str:
    for key in ("name", "title", "workout_name"):
        if payload.get(key):
            return str(payload[key])
    return source.stem.replace("_", " ").strip() or "Strength session"


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    try:
        number = float(value) if value is not None else None
        return number if number is not None and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _text(value: Any) -> str | None:
    return str(value).strip() if value not in (None, "") else None


def _first_present(values: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    return None
