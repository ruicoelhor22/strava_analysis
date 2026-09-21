from datetime import date

import pytest

from endurance_lab.analytics import (
    aerobic_decoupling,
    best_power_durations,
    first_second_halves,
    normalized_power,
    rolling_load,
    zone_seconds,
)


def test_normalized_power_is_constant_for_constant_series():
    assert normalized_power([200.0] * 120) == pytest.approx(200.0)


def test_normalized_power_requires_sufficient_coverage():
    assert normalized_power([None] * 60) is None


def test_power_duration_uses_best_rolling_average():
    values = [100.0] * 10 + [300.0] * 10 + [100.0] * 10
    result = best_power_durations(values, [5, 10, 20])
    assert result[5] == pytest.approx(300)
    assert result[10] == pytest.approx(300)
    assert result[20] == pytest.approx(200)


def test_zone_seconds_uses_time_deltas_and_boundaries():
    result = zone_seconds([100, 130, 150, 170], [0, 120, 140, 160, 200], [0, 5, 10, 15])
    assert result == [5, 5, 5, 0]


def test_first_second_halves_are_split_by_elapsed_time():
    result = first_second_halves([0, 1, 2, 3], [100, 100, 80, 80], [140, 140, 150, 150])
    assert result == (100, 80, 140, 150)


def test_decoupling_reports_efficiency_loss_for_steady_session():
    elapsed = list(range(3601))
    power = [200.0] * len(elapsed)
    hr = [140.0 if second <= 1800 else 150.0 for second in elapsed]
    result = aerobic_decoupling(elapsed, power, hr)
    assert result.status == "applicable"
    assert result.value_pct == pytest.approx(6.6667, rel=1e-3)


def test_decoupling_rejects_short_and_variable_sessions():
    short = aerobic_decoupling(list(range(100)), [200.0] * 100, [140.0] * 100)
    assert short.reason == "activity_too_short"
    elapsed = list(range(3601))
    variable = aerobic_decoupling(elapsed, [50.0 if x % 2 else 350.0 for x in elapsed], [145.0] * len(elapsed))
    assert variable.reason == "variable_or_interval_session"


def test_rolling_load_fills_rest_days_and_updates_models():
    result = rolling_load({date(2026, 1, 1): 42, date(2026, 1, 3): 84}, fitness_days=42, fatigue_days=7)
    assert len(result) == 3
    assert result[1]["load"] == 0
    assert result[0]["fitness"] == pytest.approx(1)
    assert result[0]["fatigue"] == pytest.approx(6)

