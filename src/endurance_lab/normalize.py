from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable

from endurance_lab.models import ParsedActivity, Trackpoint


def finalize_activity(activity: ParsedActivity) -> ParsedActivity:
    points = sorted(
        activity.trackpoints,
        key=lambda point: (
            point.recorded_at is None,
            point.recorded_at,
            point.sequence,
        ),
    )
    activity.trackpoints = points
    for index, point in enumerate(points):
        point.sequence = index
        if point.recorded_at is not None:
            point.elapsed_seconds = max(
                0.0, (point.recorded_at - activity.started_at).total_seconds()
            )
        if index == 0:
            continue
        previous = points[index - 1]
        delta_time = _seconds(previous, point)
        delta_distance = None
        if point.distance_m is not None and previous.distance_m is not None:
            delta_distance = max(0.0, point.distance_m - previous.distance_m)
        if point.speed_mps is None and delta_distance is not None and delta_time and delta_time > 0:
            point.speed_mps = delta_distance / delta_time
        if point.moving is None:
            threshold = 0.2 if activity.sport == "swimming" else 0.5
            if delta_time is None or delta_time <= 0 or delta_time > 30:
                point.moving = False
            elif point.speed_mps is not None:
                point.moving = point.speed_mps >= threshold
            elif delta_distance is not None:
                point.moving = delta_distance >= threshold
            else:
                point.moving = True

    times = [point.recorded_at for point in points if point.recorded_at is not None]
    if activity.ended_at is None and times:
        activity.ended_at = max(times)
    if activity.elapsed_seconds is None and activity.ended_at is not None:
        activity.elapsed_seconds = max(
            0.0, (activity.ended_at - activity.started_at).total_seconds()
        )
    if activity.elapsed_seconds is None:
        activity.elapsed_seconds = _sum(lap.duration_seconds for lap in activity.laps)
    if activity.moving_seconds is None:
        activity.moving_seconds = _moving_seconds(points)
        if not activity.moving_seconds and activity.sport in {"strength", "swimming", "other"}:
            activity.moving_seconds = activity.elapsed_seconds

    distances = [point.distance_m for point in points if point.distance_m is not None]
    if activity.distance_m is None:
        activity.distance_m = max(distances) if distances else _sum(
            lap.distance_m for lap in activity.laps
        )
    altitude = [point.altitude_m for point in points if point.altitude_m is not None]
    gain, loss = elevation_change(altitude)
    if activity.ascent_m is None:
        activity.ascent_m = gain
    if activity.descent_m is None:
        activity.descent_m = loss
    if activity.calories is None:
        activity.calories = _sum(lap.calories for lap in activity.laps)

    for average_field, maximum_field, point_field in (
        ("avg_hr", "max_hr", "heart_rate"),
        ("avg_cadence", "max_cadence", "cadence"),
        ("avg_power_w", "max_power_w", "power_w"),
        ("avg_speed_mps", "max_speed_mps", "speed_mps"),
    ):
        values = [getattr(point, point_field) for point in points]
        if getattr(activity, average_field) is None:
            setattr(activity, average_field, mean(values))
        if getattr(activity, maximum_field) is None:
            setattr(activity, maximum_field, maximum(values))
        if getattr(activity, average_field) is None:
            setattr(activity, average_field, mean(getattr(lap, average_field) for lap in activity.laps))
        if getattr(activity, maximum_field) is None:
            setattr(activity, maximum_field, maximum(getattr(lap, maximum_field) for lap in activity.laps))
    return activity


def activity_quality(activity: ParsedActivity) -> tuple[int, dict[str, int | bool]]:
    points = activity.trackpoints
    count = len(points)
    coverage = {
        "trackpoints": count,
        "hr_points": sum(point.heart_rate is not None for point in points),
        "gps_points": sum(point.latitude is not None and point.longitude is not None for point in points),
        "elevation_points": sum(point.altitude_m is not None for point in points),
        "cadence_points": sum(point.cadence is not None for point in points),
        "power_points": sum(point.power_w is not None for point in points),
        "speed_points": sum(point.speed_mps is not None for point in points),
        "laps": len(activity.laps),
        "strength_sets": len(activity.strength_sets),
        "has_distance": activity.distance_m is not None and activity.distance_m > 0,
    }
    score = min(count, 3600) // 90
    score += 18 if coverage["power_points"] else 0
    score += 14 if coverage["hr_points"] else 0
    score += 10 if coverage["gps_points"] else 0
    score += 8 if coverage["elevation_points"] else 0
    score += 7 if coverage["cadence_points"] else 0
    score += 5 if coverage["speed_points"] else 0
    score += min(int(coverage["laps"]), 10)
    score += min(int(coverage["strength_sets"]), 20)
    score += 3 if coverage["has_distance"] else 0
    return int(score), coverage


def elevation_change(values: list[float]) -> tuple[float | None, float | None]:
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


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return fmean(clean) if clean else None


def maximum(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(clean) if clean else None


def _sum(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) if clean else None


def _seconds(before: Trackpoint, after: Trackpoint) -> float | None:
    if before.recorded_at is None or after.recorded_at is None:
        return None
    return (after.recorded_at - before.recorded_at).total_seconds()


def _moving_seconds(points: list[Trackpoint]) -> float:
    total = 0.0
    for before, after in zip(points, points[1:]):
        delta = _seconds(before, after)
        if after.moving and delta and 0 < delta <= 30:
            total += delta
    return total
