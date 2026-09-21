from __future__ import annotations

import hashlib
import math
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from endurance_lab.models import Activity, Lap, Trackpoint
from endurance_lab.identity import source_activity_id_from_filename
from endurance_lab.normalize import elevation_change, finalize_activity


SPORTS = {
    "biking": "cycling",
    "cycling": "cycling",
    "running": "running",
    "swimming": "swimming",
    "other": "other",
    "multisport": "other",
}


def parse_tcx(path: str | Path, file_sha256: str | None = None) -> list[Activity]:
    source = Path(path)
    sha256 = file_sha256 or hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid TCX XML in {source.name}: {exc}") from exc

    activity_nodes = _descendants(root, "Activity")
    source_id = source_activity_id_from_filename(source.name) if len(activity_nodes) == 1 else None
    parsed: list[Activity] = []
    for index, node in enumerate(activity_nodes):
        activity_id_text = _text(_first_child(node, "Id"))
        strava_id = source_id or _numeric_activity_id(activity_id_text)
        stable_id = f"strava:{strava_id}" if strava_id else f"tcx:{sha256}:{index}"
        laps = _children(node, "Lap")
        points: list[Trackpoint] = []
        parsed_laps: list[Lap] = []
        for lap_number, lap_node in enumerate(laps, start=1):
            start_sequence = len(points)
            for point_node in _descendants(lap_node, "Trackpoint"):
                points.append(_parse_trackpoint(point_node, len(points)))
            parsed_laps.append(_parse_lap(lap_node, lap_number, points[start_sequence:]))

        started_at = _dt(activity_id_text)
        if started_at is None:
            started_at = next((p.recorded_at for p in points if p.recorded_at), None)
        if started_at is None:
            raise ValueError(f"Activity {index + 1} in {source.name} has no timestamp")
        creator = _first_child(node, "Creator")
        notes = _text(_first_child(node, "Notes"))
        activity = Activity(
            stable_id=stable_id,
            source_activity_id=strava_id,
            sport=_sport(node.attrib.get("Sport"), notes, source.name),
            started_at=started_at,
            name=notes,
            device=_device_name(creator),
            laps=parsed_laps,
            trackpoints=points,
        )
        finalize_activity(activity)
        parsed.append(activity)
    if not parsed:
        raise ValueError(f"No Activity elements found in {source.name}")
    return parsed


def _parse_trackpoint(node: ET.Element, sequence: int) -> Trackpoint:
    position = _first_child(node, "Position")
    return Trackpoint(
        sequence=sequence,
        recorded_at=_dt(_text(_first_child(node, "Time"))),
        distance_m=_number(_text(_first_child(node, "DistanceMeters"))),
        latitude=_number(_text(_first_child(position, "LatitudeDegrees"))),
        longitude=_number(_text(_first_child(position, "LongitudeDegrees"))),
        altitude_m=_number(_text(_first_child(node, "AltitudeMeters"))),
        heart_rate=_number(_text(_first_descendant(node, "Value", parent_name="HeartRateBpm"))),
        cadence=_number(_text(_first_child(node, "Cadence")))
        or _number(_text(_first_descendant(node, "RunCadence"))),
        speed_mps=_number(_text(_first_descendant(node, "Speed"))),
        power_w=_number(_text(_first_descendant(node, "Watts"))),
        temperature_c=_number(_text(_first_descendant(node, "Temp"))),
    )


def _parse_lap(node: ET.Element, number: int, points: list[Trackpoint]) -> Lap:
    started = _dt(node.attrib.get("StartTime"))
    duration = _number(_text(_first_child(node, "TotalTimeSeconds")))
    ended = None
    if points:
        ended = next((p.recorded_at for p in reversed(points) if p.recorded_at), None)
    elevation = elevation_change([point.altitude_m for point in points if point.altitude_m is not None])
    lap = Lap(
        lap_number=number,
        started_at=started,
        ended_at=ended,
        duration_seconds=duration,
        distance_m=_number(_text(_first_child(node, "DistanceMeters"))),
        calories=_number(_text(_first_child(node, "Calories"))),
        avg_hr=_number(_text(_first_descendant(node, "Value", parent_name="AverageHeartRateBpm"))),
        max_hr=_number(_text(_first_descendant(node, "Value", parent_name="MaximumHeartRateBpm"))),
        avg_speed_mps=_number(_text(_first_descendant(node, "AvgSpeed"))),
        max_speed_mps=_number(_text(_first_child(node, "MaximumSpeed"))),
        avg_cadence=_number(_text(_first_child(node, "Cadence"))),
        max_cadence=_number(_text(_first_descendant(node, "MaxBikeCadence"))),
        avg_power_w=_number(_text(_first_descendant(node, "AvgWatts"))),
        max_power_w=_number(_text(_first_descendant(node, "MaxWatts"))),
    )
    lap.ascent_m, lap.descent_m = elevation
    for field_name, values in (
        ("avg_hr", [point.heart_rate for point in points]),
        ("avg_speed_mps", [point.speed_mps for point in points]),
        ("avg_cadence", [point.cadence for point in points]),
        ("avg_power_w", [point.power_w for point in points]),
    ):
        if getattr(lap, field_name) is None:
            setattr(lap, field_name, _mean(values))
    for field_name, values in (
        ("max_hr", [point.heart_rate for point in points]),
        ("max_speed_mps", [point.speed_mps for point in points]),
        ("max_cadence", [point.cadence for point in points]),
        ("max_power_w", [point.power_w for point in points]),
    ):
        if getattr(lap, field_name) is None:
            setattr(lap, field_name, _max(values))
    return lap


def _add_elapsed_and_moving(points: list[Trackpoint], started_at: datetime) -> None:
    for index, point in enumerate(points):
        if point.recorded_at:
            point.elapsed_seconds = max(0.0, (point.recorded_at - started_at).total_seconds())
        if index == 0:
            point.moving = None
            continue
        previous = points[index - 1]
        delta_time = _seconds_between(previous.recorded_at, point.recorded_at)
        delta_distance = None
        if point.distance_m is not None and previous.distance_m is not None:
            delta_distance = max(0.0, point.distance_m - previous.distance_m)
        speed = point.speed_mps
        if speed is None and delta_distance is not None and delta_time and delta_time > 0:
            speed = delta_distance / delta_time
            point.speed_mps = speed
        if delta_time is None or delta_time <= 0 or delta_time > 30:
            point.moving = False
        elif speed is not None:
            point.moving = speed >= 0.5
        elif delta_distance is not None:
            point.moving = delta_distance >= 0.5
        else:
            point.moving = True


def _summarize_activity(activity: Activity) -> None:
    points = activity.trackpoints
    times = [p.recorded_at for p in points if p.recorded_at]
    activity.ended_at = max(times) if times else _latest_lap_end(activity.laps)
    if activity.ended_at:
        activity.elapsed_seconds = max(0.0, (activity.ended_at - activity.started_at).total_seconds())
    if activity.elapsed_seconds is None:
        activity.elapsed_seconds = _sum_values(lap.duration_seconds for lap in activity.laps)
    activity.moving_seconds = _moving_seconds(points)
    if activity.moving_seconds == 0 and activity.sport in {"strength", "swimming", "other"}:
        activity.moving_seconds = activity.elapsed_seconds

    distances = [p.distance_m for p in points if p.distance_m is not None]
    activity.distance_m = max(distances) if distances else _sum_values(l.distance_m for l in activity.laps)
    altitude = [p.altitude_m for p in points if p.altitude_m is not None]
    activity.ascent_m, activity.descent_m = _elevation_change(altitude)
    activity.calories = _sum_values(l.calories for l in activity.laps)

    activity.avg_hr, activity.max_hr = _avg_max(p.heart_rate for p in points)
    activity.avg_cadence, activity.max_cadence = _avg_max(p.cadence for p in points)
    activity.avg_power_w, activity.max_power_w = _avg_max(p.power_w for p in points)
    activity.avg_speed_mps, activity.max_speed_mps = _avg_max(p.speed_mps for p in points)
    for field_name in ("avg_hr", "max_hr", "avg_cadence", "max_cadence", "avg_power_w", "max_power_w", "avg_speed_mps", "max_speed_mps"):
        if getattr(activity, field_name) is None:
            values = [getattr(lap, field_name) for lap in activity.laps]
            setattr(activity, field_name, _mean(values) if field_name.startswith("avg_") else _max(values))


def _moving_seconds(points: list[Trackpoint]) -> float:
    total = 0.0
    for previous, point in zip(points, points[1:]):
        delta = _seconds_between(previous.recorded_at, point.recorded_at)
        if point.moving and delta and 0 < delta <= 30:
            total += delta
    return total


def _elevation_change(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    gain = loss = 0.0
    for before, after in zip(values, values[1:]):
        delta = after - before
        if abs(delta) > 25:
            continue
        if delta > 0:
            gain += delta
        else:
            loss -= delta
    return gain, loss


def _sport(value: str | None, notes: str | None = None, filename: str | None = None) -> str:
    normalized = (value or "other").strip().lower()
    if normalized in SPORTS and SPORTS[normalized] != "other":
        return SPORTS[normalized]
    if "bike" in normalized or "ride" in normalized:
        return "cycling"
    if "run" in normalized:
        return "running"
    if "swim" in normalized:
        return "swimming"
    if "weight" in normalized or "strength" in normalized:
        return "strength"
    hint = unicodedata.normalize("NFKD", f"{notes or ''} {filename or ''}").encode("ascii", "ignore").decode().lower()
    if any(keyword in hint for keyword in ("swim", "natacao", "nadar")):
        return "swimming"
    if any(keyword in hint for keyword in ("strength", "weight training", "weights", "gym", "forca")):
        return "strength"
    return SPORTS.get(normalized, "other")


def _numeric_activity_id(value: str | None) -> str | None:
    return value if value and value.isdigit() and len(value) >= 7 else None


def _device_name(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    name = _text(_first_child(node, "Name"))
    unit = _text(_first_child(node, "UnitId"))
    return " / ".join(part for part in (name, unit) if part) or None


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _number(value: str | None) -> float | None:
    try:
        number = float(value) if value not in (None, "") else None
        return number if number is not None and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _text(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    stripped = node.text.strip()
    return stripped or None


def _local(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _children(node: ET.Element | None, name: str) -> list[ET.Element]:
    return [] if node is None else [child for child in list(node) if _local(child) == name]


def _first_child(node: ET.Element | None, name: str) -> ET.Element | None:
    return next(iter(_children(node, name)), None)


def _descendants(node: ET.Element, name: str) -> list[ET.Element]:
    return [element for element in node.iter() if _local(element) == name]


def _first_descendant(
    node: ET.Element, name: str, parent_name: str | None = None
) -> ET.Element | None:
    for parent in node.iter():
        if parent_name and _local(parent) != parent_name:
            continue
        for child in parent.iter():
            if _local(child) == name:
                return child
    return None


def _seconds_between(before: datetime | None, after: datetime | None) -> float | None:
    return None if before is None or after is None else (after - before).total_seconds()


def _avg_max(values) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    return ((_mean(clean), max(clean)) if clean else (None, None))


def _mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _max(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _sum_values(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) if clean else None


def _latest_lap_end(laps: list[Lap]) -> datetime | None:
    values = [lap.ended_at for lap in laps if lap.ended_at]
    return max(values) if values else None
