from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from garmin_fit_sdk import Decoder, Stream

from endurance_lab.identity import source_activity_id_from_filename
from endurance_lab.models import Lap, ParsedActivity, Trackpoint
from endurance_lab.normalize import finalize_activity


SEMICIRCLE_TO_DEGREES = 180.0 / (2**31)


def parse_fit(path: str | Path, file_sha256: str | None = None) -> list[ParsedActivity]:
    source = Path(path)
    sha256 = file_sha256 or hashlib.sha256(source.read_bytes()).hexdigest()
    stream = None
    try:
        stream = Stream.from_file(str(source))
        messages, errors = Decoder(stream).read()
    except Exception as exc:
        raise ValueError(f"Invalid FIT file {source.name}: {exc}") from exc
    finally:
        if stream is not None:
            stream.close()

    sessions = list(messages.get("session_mesgs", []))
    records = list(messages.get("record_mesgs", []))
    lap_messages = list(messages.get("lap_mesgs", []))
    file_id = _first(messages.get("file_id_mesgs", []))
    if not sessions:
        sessions = [_synthetic_session(records, file_id)]
    if not records and not sessions[0].get("start_time"):
        reason = "; ".join(str(error) for error in errors) or "no sessions or records"
        raise ValueError(f"FIT file contains no activity data: {reason}")

    filename_id = source_activity_id_from_filename(source.name) if len(sessions) == 1 else None
    parsed: list[ParsedActivity] = []
    for index, session in enumerate(sessions):
        start = _datetime(session.get("start_time")) or _first_timestamp(records) or _datetime(file_id.get("time_created"))
        if start is None:
            raise ValueError(f"FIT session {index + 1} in {source.name} has no timestamp")
        end = _datetime(session.get("timestamp")) or _last_timestamp(records)
        session_records = _between(records, start, end)
        session_laps = _between(lap_messages, start, end, key="start_time")
        sport = _sport(session.get("sport"), session.get("sub_sport"))
        points = [_record_to_point(record, sequence) for sequence, record in enumerate(session_records)]
        laps = [_lap(message, number) for number, message in enumerate(session_laps, start=1)]
        source_id = filename_id
        activity = ParsedActivity(
            stable_id=f"strava:{source_id}" if source_id else f"fit:{sha256}:{index}",
            source_activity_id=source_id,
            sport=sport,
            started_at=start,
            ended_at=end,
            name=_text(session.get("sport_profile_name")),
            elapsed_seconds=_number(session.get("total_elapsed_time")),
            moving_seconds=_first_number(session, "total_timer_time", "active_time"),
            distance_m=_number(session.get("total_distance")),
            ascent_m=_number(session.get("total_ascent")),
            descent_m=_number(session.get("total_descent")),
            calories=_number(session.get("total_calories")),
            avg_hr=_number(session.get("avg_heart_rate")),
            max_hr=_number(session.get("max_heart_rate")),
            avg_speed_mps=_prefer(session, "enhanced_avg_speed", "avg_speed"),
            max_speed_mps=_prefer(session, "enhanced_max_speed", "max_speed"),
            avg_cadence=_cadence(session, "avg_cadence", "fractional_avg_cadence"),
            max_cadence=_cadence(session, "max_cadence", "fractional_max_cadence"),
            avg_power_w=_number(session.get("avg_power")),
            max_power_w=_number(session.get("max_power")),
            device=_device_name(messages),
            source_metadata=_source_metadata(messages, session, errors),
            laps=laps,
            trackpoints=points,
        )
        parsed.append(finalize_activity(activity))
    return parsed


def semicircles_to_degrees(value: int | float | None) -> float | None:
    number = _number(value)
    if number is None:
        return None
    degrees = number * SEMICIRCLE_TO_DEGREES
    return degrees if -180 <= degrees <= 180 else None


def _record_to_point(record: dict[str, Any], sequence: int) -> Trackpoint:
    return Trackpoint(
        sequence=sequence,
        recorded_at=_datetime(record.get("timestamp")),
        distance_m=_number(record.get("distance")),
        latitude=semicircles_to_degrees(record.get("position_lat")),
        longitude=semicircles_to_degrees(record.get("position_long")),
        altitude_m=_prefer(record, "enhanced_altitude", "altitude"),
        heart_rate=_number(record.get("heart_rate")),
        cadence=_cadence(record, "cadence", "fractional_cadence"),
        speed_mps=_prefer(record, "enhanced_speed", "speed"),
        power_w=_number(record.get("power")),
        temperature_c=_number(record.get("temperature")),
    )


def _lap(message: dict[str, Any], number: int) -> Lap:
    return Lap(
        lap_number=number,
        started_at=_datetime(message.get("start_time")),
        ended_at=_datetime(message.get("timestamp")),
        duration_seconds=_first_number(message, "total_timer_time", "total_elapsed_time"),
        distance_m=_number(message.get("total_distance")),
        ascent_m=_number(message.get("total_ascent")),
        descent_m=_number(message.get("total_descent")),
        calories=_number(message.get("total_calories")),
        avg_hr=_number(message.get("avg_heart_rate")),
        max_hr=_number(message.get("max_heart_rate")),
        avg_speed_mps=_prefer(message, "enhanced_avg_speed", "avg_speed"),
        max_speed_mps=_prefer(message, "enhanced_max_speed", "max_speed"),
        avg_cadence=_cadence(message, "avg_cadence", "fractional_avg_cadence"),
        max_cadence=_cadence(message, "max_cadence", "fractional_max_cadence"),
        avg_power_w=_number(message.get("avg_power")),
        max_power_w=_number(message.get("max_power")),
    )


def _source_metadata(messages, session, errors) -> dict[str, Any]:
    file_id = _first(messages.get("file_id_mesgs", []))
    devices = []
    for item in messages.get("device_info_mesgs", []):
        devices.append(
            _clean_dict(
                item,
                ("manufacturer", "product", "garmin_product", "device_type", "software_version", "hardware_version", "source_type"),
            )
        )
    return {
        "fit_profile": _clean_dict(file_id, ("type", "manufacturer", "product", "garmin_product", "time_created")),
        "sub_sport": session.get("sub_sport"),
        "sport_profile_name": session.get("sport_profile_name"),
        "devices": [device for device in devices if device],
        "decoder_errors": [str(error) for error in errors],
        "unknown_message_types": sorted(str(key) for key in messages if str(key)[0].isdigit()),
    }


def _device_name(messages) -> str | None:
    file_id = _first(messages.get("file_id_mesgs", []))
    manufacturer = _text(file_id.get("manufacturer"))
    product = _text(file_id.get("garmin_product")) or _text(file_id.get("product"))
    return " / ".join(value for value in (manufacturer, product) if value) or None


def _synthetic_session(records, file_id) -> dict[str, Any]:
    return {
        "start_time": _first_timestamp(records) or _datetime(file_id.get("time_created")),
        "timestamp": _last_timestamp(records),
        "sport": "generic",
    }


def _between(messages, start, end, key: str = "timestamp"):
    if not messages:
        return []
    selected = []
    for message in messages:
        timestamp = _datetime(message.get(key)) or _datetime(message.get("timestamp"))
        if timestamp is None:
            continue
        if timestamp >= start and (end is None or timestamp <= end):
            selected.append(message)
    return selected


def _first_timestamp(records) -> datetime | None:
    values = [_datetime(record.get("timestamp")) for record in records]
    clean = [value for value in values if value is not None]
    return min(clean) if clean else None


def _last_timestamp(records) -> datetime | None:
    values = [_datetime(record.get("timestamp")) for record in records]
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _sport(sport: Any, sub_sport: Any) -> str:
    text = f"{sport or ''} {sub_sport or ''}".lower()
    if "cycl" in text or "bik" in text:
        return "cycling"
    if "run" in text:
        return "running"
    if "swim" in text:
        return "swimming"
    if "strength" in text or "training" in text:
        return "strength"
    return "other"


def _prefer(values: dict[str, Any], preferred: str, fallback: str) -> float | None:
    value = _number(values.get(preferred))
    return value if value is not None else _number(values.get(fallback))


def _first_number(values: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(values.get(key))
        if value is not None:
            return value
    return None


def _cadence(values: dict[str, Any], whole_key: str, fractional_key: str) -> float | None:
    whole = _number(values.get(whole_key))
    fractional = _number(values.get(fractional_key))
    if whole is None:
        return fractional
    return whole + (fractional or 0.0)


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value) if value is not None else None
        return number if number is not None and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _first(values) -> dict[str, Any]:
    return dict(values[0]) if values else {}


def _clean_dict(values: dict[str, Any], keys) -> dict[str, Any]:
    return {key: str(values[key]) if isinstance(values.get(key), datetime) else values[key] for key in keys if values.get(key) is not None}
