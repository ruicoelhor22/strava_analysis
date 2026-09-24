from unittest.mock import patch

from dashboard import app
from dashboard._shared import zone_range_labels
from endurance_lab.config import ftp_at


def test_zone_labels_show_integer_heart_rate_boundaries():
    assert zone_range_labels([0, 125, 145, 161, 177, 191, 208], "bpm") == [
        "Z1 · 0–124 bpm",
        "Z2 · 125–144 bpm",
        "Z3 · 145–160 bpm",
        "Z4 · 161–176 bpm",
        "Z5 · 177–190 bpm",
        "Z6 · 191–207 bpm",
    ]


def test_power_zone_labels_use_activity_ftp():
    boundaries = [percentage * 285 for percentage in (0, .55, .75, .90, 1.05, 1.20, 10)]
    assert zone_range_labels(boundaries, "W", open_last=True) == [
        "Z1 · 0–156 W",
        "Z2 · 157–213 W",
        "Z3 · 214–256 W",
        "Z4 · 257–299 W",
        "Z5 · 300–341 W",
        "Z6 · 342+ W",
    ]


def test_new_ftp_keeps_historical_reference():
    config = {"cycling": {"ftp_history": [
        {"effective_from": "2020-01-01", "watts": 265},
        {"effective_from": "2026-09-24", "watts": 285},
    ]}}
    assert ftp_at(config, "2026-09-23") == 265
    assert ftp_at(config, "2026-09-24") == 285


def test_activity_power_chart_uses_recorded_ftp_not_current_ftp():
    config = {"cycling": {"power_zone_percentages": [0, .55, .75, .90, 1.05, 1.20, 10]}}
    record = {"ftp_w": 265, "power_zones_json": "[60, 120, 180, 240, 300, 360]"}
    with (
        patch.object(app, "load_athlete_config", return_value=config),
        patch.object(app.st, "subheader"),
        patch.object(app.st, "caption") as caption,
        patch.object(app.st, "plotly_chart") as plot,
    ):
        app._zones_panel(record, "power_zones_json", "Power zones", "#73E0A9")
    figure = plot.call_args.args[0]
    assert figure.data[0].orientation == "h"
    assert figure.data[0].y[1] == "Z2 · 146–198 W"
    caption.assert_any_call("Based on the 265 W FTP used for this activity.")
