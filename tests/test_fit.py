from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from garmin_fit_sdk import Encoder, Profile

from endurance_lab.analytics import analyze_database
from endurance_lab.db import connect
from endurance_lab.fit import parse_fit, semicircles_to_degrees
from endurance_lab.importer import import_file


def test_fit_parser_extracts_units_coordinates_power_hr_cadence_and_laps():
    source = _write_fit(_test_dir() / "987654321.fit", seconds=120)
    ride = parse_fit(source)[0]
    assert ride.source_activity_id == "987654321"
    assert ride.sport == "cycling"
    assert ride.elapsed_seconds == pytest.approx(120)
    assert ride.moving_seconds == pytest.approx(120)
    assert ride.distance_m == pytest.approx(1200)
    assert ride.avg_power_w == pytest.approx(200)
    assert ride.max_power_w == pytest.approx(200)
    assert ride.avg_hr == pytest.approx(145)
    assert ride.avg_cadence == pytest.approx(86)
    assert len(ride.laps) == 2
    assert len(ride.trackpoints) == 121
    assert ride.trackpoints[0].latitude == pytest.approx(1.0, abs=1e-6)
    assert ride.trackpoints[0].longitude == pytest.approx(2.0, abs=1e-6)
    assert ride.trackpoints[0].speed_mps == pytest.approx(10.0)
    assert ride.trackpoints[0].altitude_m == pytest.approx(100.0)
    assert ride.trackpoints[0].power_w == 200
    assert ride.trackpoints[0].heart_rate == 140
    assert ride.trackpoints[0].cadence == pytest.approx(86.5)
    assert ride.trackpoints[0].temperature_c == 20


def test_fit_semicircle_conversion():
    assert semicircles_to_degrees(2**30) == pytest.approx(90.0)
    assert semicircles_to_degrees(-(2**29)) == pytest.approx(-45.0)


def test_fit_cycling_flows_through_existing_analytics(monkeypatch):
    root = _test_dir()
    source = _write_fit(root / "987654322.fit", seconds=3600)
    database = root / "analytics.sqlite3"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    result = import_file(source, database)
    assert result.status == "imported"
    assert analyze_database(database) == 1
    with connect(database) as connection:
        metric = connection.execute("SELECT * FROM derived_activity_metrics").fetchone()
        curve_count = connection.execute("SELECT COUNT(*) FROM power_curve_results").fetchone()[0]
    assert metric["normalized_power_w"] == pytest.approx(200, rel=0.01)
    assert metric["estimated_tss"] is not None
    assert metric["efficiency_factor"] is not None
    assert metric["first_half_output"] is not None
    assert metric["second_half_output"] is not None
    assert metric["late_fade_pct"] == pytest.approx(0.0, abs=0.01)
    assert metric["aerobic_decoupling_pct"] == pytest.approx(6.67, rel=0.02)
    assert metric["hr_zones_json"] != "[]"
    assert metric["power_zones_json"] != "[]"
    assert curve_count == 10


def test_fit_running_uses_normalized_speed_and_hr(monkeypatch):
    root = _test_dir()
    source = _write_fit(root / "987654323.fit", sport="running", seconds=900)
    database = root / "running.sqlite3"
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    import_file(source, database)
    analyze_database(database)
    with connect(database) as connection:
        activity = connection.execute("SELECT * FROM activities").fetchone()
        metric = connection.execute("SELECT * FROM derived_activity_metrics").fetchone()
    assert activity["sport"] == "running"
    assert metric["pace_seconds_per_km"] == pytest.approx(333.333, rel=0.02)
    assert metric["efficiency_factor"] is not None


def test_malformed_fit_is_reported(monkeypatch):
    root = _test_dir()
    source = root / "broken.fit"
    source.write_bytes(b"not-a-fit-file")
    monkeypatch.setenv("ENDURANCE_RAW_DIR", str(root / "raw"))
    result = import_file(source, root / "broken.sqlite3")
    assert result.status == "failed"
    assert "FIT" in (result.message or "")


def _write_fit(path: Path, sport: str = "cycling", seconds: int = 120) -> Path:
    encoder = Encoder()
    start = datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc)
    speed = 10.0 if sport == "cycling" else 3.0
    distance = speed * seconds
    encoder.on_mesg(
        Profile["mesg_num"]["FILE_ID"],
        {
            "type": "activity",
            "manufacturer": "development",
            "product": 1,
            "time_created": start,
        },
    )
    for second in range(seconds + 1):
        record = {
            "timestamp": start + timedelta(seconds=second),
            "position_lat": int((1.0 + second / 10_000_000) * 2**31 / 180),
            "position_long": int((2.0 + second / 10_000_000) * 2**31 / 180),
            "distance": speed * second,
            "heart_rate": 140 if second <= seconds / 2 else 150,
            "cadence": 86,
            "fractional_cadence": 0.5,
            "enhanced_speed": speed,
            "enhanced_altitude": 100.0 + second / 100,
            "temperature": 20,
        }
        if sport == "cycling":
            record["power"] = 200
        encoder.on_mesg(Profile["mesg_num"]["RECORD"], record)
    half = seconds // 2
    for index, lap_start in enumerate((0, half)):
        lap_end = half if index == 0 else seconds
        lap = {
            "timestamp": start + timedelta(seconds=lap_end),
            "start_time": start + timedelta(seconds=lap_start),
            "total_elapsed_time": float(lap_end - lap_start),
            "total_timer_time": float(lap_end - lap_start),
            "total_distance": speed * (lap_end - lap_start),
            "sport": sport,
            "avg_heart_rate": 140 if index == 0 else 150,
            "max_heart_rate": 150,
            "avg_cadence": 86,
            "enhanced_avg_speed": speed,
            "enhanced_max_speed": speed,
        }
        if sport == "cycling":
            lap.update(avg_power=200, max_power=200)
        encoder.on_mesg(Profile["mesg_num"]["LAP"], lap)
    session = {
        "timestamp": start + timedelta(seconds=seconds),
        "start_time": start,
        "total_elapsed_time": float(seconds),
        "total_timer_time": float(seconds),
        "total_distance": distance,
        "sport": sport,
        "sub_sport": "road" if sport == "cycling" else "generic",
        "num_laps": 2,
        "avg_heart_rate": 145,
        "max_heart_rate": 150,
        "avg_cadence": 86,
        "max_cadence": 86,
        "enhanced_avg_speed": speed,
        "enhanced_max_speed": speed,
        "total_ascent": float(seconds / 100),
        "total_descent": 0.0,
        "total_calories": 500,
    }
    if sport == "cycling":
        session.update(avg_power=200, max_power=200)
    encoder.on_mesg(Profile["mesg_num"]["SESSION"], session)
    path.write_bytes(encoder.close())
    return path


def _test_dir() -> Path:
    path = Path("data/test-runs") / uuid4().hex
    path.mkdir(parents=True)
    return path
