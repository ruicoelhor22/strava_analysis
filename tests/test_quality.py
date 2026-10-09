from __future__ import annotations

from datetime import datetime, timedelta, timezone

from endurance_lab.analytics import _sanitized_sensors, derive_activity
from endurance_lab.quality import _issues


class _GpsConnection:
    def __init__(self, minutes: int):
        start = datetime(2026, 9, 24, tzinfo=timezone.utc)
        self.rows = [
            {"recorded_at": start.isoformat(), "latitude": 40.0, "longitude": -8.0},
            {"recorded_at": (start + timedelta(minutes=minutes)).isoformat(), "latitude": 40.01, "longitude": -8.01},
        ]

    def execute(self, *_args):
        return self

    def fetchall(self):
        return self.rows


def _activity(moving_seconds, sport="cycling"):
    return {"id": 1, "sport": sport, "elapsed_seconds": 3600,
            "moving_seconds": moving_seconds, "distance_m": 20_000,
            "started_at": "2026-09-24T08:00:00+00:00", "metadata_only": 0,
            "max_hr": None, "max_power_w": None}


def _stream(**overrides):
    return {"points": 3600, "hr_points": 0, "gps_points": 2,
            "first_time": None, "last_time": None, "stream_min_hr": None,
            "stream_max_hr": None, "stream_max_power": None,
            "stream_max_speed": None, "stream_max_cadence": None, **overrides}


def test_pause_is_info_not_a_gps_warning():
    issues = _issues(_GpsConnection(20), _activity(1800), _stream())
    assert [(x["code"], x["severity"]) for x in issues] == [("recording_pause", "info")]


def test_unexplained_substantial_gap_is_warning():
    issues = _issues(_GpsConnection(25), _activity(3600), _stream())
    assert any(x["code"] == "substantial_gps_gap" and x["severity"] == "warning" for x in issues)


def test_strength_without_endurance_sensors_is_not_warned():
    issues = _issues(_GpsConnection(0), _activity(1800, "strength"), _stream(points=0, gps_points=0))
    assert issues == []


def test_invalid_hr_is_excluded_from_analytics_without_changing_input():
    power = [200, 200, 4000, 200]
    hr = [130, 210, 132, 260]
    speed = [5, 50, 5, 5]
    clean_power, clean_hr, clean_speed = _sanitized_sensors(power, hr, speed)
    assert clean_power == [200, 200, None, 200]
    assert clean_hr == [130, None, 132, None]
    assert clean_speed == [5, None, 5, 5]
    assert hr == [130, 210, 132, 260]


def test_metadata_only_has_no_stream_derived_metrics():
    metrics, best = derive_activity(
        {"sport": "cycling", "metadata_only": 1, "elapsed_seconds": 1800,
         "moving_seconds": 1800, "distance_m": 0, "started_at": "2026-09-24T08:00:00+00:00"},
        [], {"analysis": {}, "heart_rate": {}, "cycling": {}},
    )
    assert metrics["estimated_hr_load"] is None
    assert metrics["aerobic_decoupling_pct"] is None
    assert metrics["load_method"] == "duration_only_estimate"
    assert best == {}
