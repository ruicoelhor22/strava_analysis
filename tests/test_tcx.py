from pathlib import Path
from uuid import uuid4

from endurance_lab.tcx import parse_tcx


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_ride.tcx"


def test_parser_extracts_activity_lap_and_extension_streams():
    activities = parse_tcx(FIXTURE)
    assert len(activities) == 1
    ride = activities[0]
    assert ride.sport == "cycling"
    assert ride.distance_m == 900
    assert ride.elapsed_seconds == 90
    assert ride.moving_seconds == 90
    assert ride.ascent_m == 12
    assert ride.descent_m == 2
    assert ride.avg_power_w == 201.25
    assert ride.max_hr == 151
    assert ride.device == "Synthetic Computer / 000000"
    assert len(ride.laps) == 1
    assert len(ride.trackpoints) == 4
    assert ride.trackpoints[1].speed_mps == 10


def test_parser_handles_missing_optional_fields():
    tmp_path = _test_dir()
    source = tmp_path / "minimal.tcx"
    source.write_text(
        """<TrainingCenterDatabase><Activities><Activity Sport="Running">
        <Id>2026-02-01T10:00:00Z</Id><Lap StartTime="2026-02-01T10:00:00Z">
        <TotalTimeSeconds>60</TotalTimeSeconds><Track><Trackpoint>
        <Time>2026-02-01T10:00:00Z</Time></Trackpoint><Trackpoint>
        <Time>2026-02-01T10:01:00Z</Time></Trackpoint></Track></Lap>
        </Activity></Activities></TrainingCenterDatabase>""",
        encoding="utf-8",
    )
    run = parse_tcx(source)[0]
    assert run.sport == "running"
    assert run.distance_m is None
    assert run.avg_hr is None
    assert run.avg_power_w is None


def test_parser_uses_filename_hint_for_other_sport():
    source = _test_dir() / "Natacao_Aguas_Abertas.tcx"
    source.write_text(
        """<TrainingCenterDatabase><Activities><Activity Sport="Other">
        <Id>2026-02-01T10:00:00Z</Id><Lap StartTime="2026-02-01T10:00:00Z">
        <TotalTimeSeconds>60</TotalTimeSeconds><Track><Trackpoint>
        <Time>2026-02-01T10:00:00Z</Time></Trackpoint><Trackpoint>
        <Time>2026-02-01T10:01:00Z</Time></Trackpoint></Track></Lap>
        </Activity></Activities></TrainingCenterDatabase>""",
        encoding="utf-8",
    )
    assert parse_tcx(source)[0].sport == "swimming"


def _test_dir() -> Path:
    path = Path("data/test-runs") / uuid4().hex
    path.mkdir(parents=True)
    return path
