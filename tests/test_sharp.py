import pytest

from slam.sharp import PLACEHOLDER_TABLE, SharpCalibration


def test_interpolates_between_points():
    cal = SharpCalibration([(400, 0.20), (300, 0.30), (200, 0.50)])
    assert cal.distance(400) == pytest.approx(0.20)
    assert cal.distance(350) == pytest.approx(0.25)
    assert cal.distance(250) == pytest.approx(0.40)
    assert (cal.min_m, cal.max_m) == (0.20, 0.50)


def test_outside_table_is_none():
    cal = SharpCalibration([(400, 0.20), (200, 0.50)])
    assert cal.distance(401) is None
    assert cal.distance(199) is None
    assert cal.distance(None) is None


def test_rejects_non_monotonic_tables():
    with pytest.raises(ValueError):
        SharpCalibration([(400, 0.20), (300, 0.30), (350, 0.40)])
    with pytest.raises(ValueError):
        SharpCalibration([(400, 0.20)])


def test_save_load_round_trip(tmp_path):
    path = str(tmp_path / "cal" / "sharp.json")
    SharpCalibration(PLACEHOLDER_TABLE).save(path)
    cal = SharpCalibration.load(path)
    assert cal.distance(285) == pytest.approx(0.30)
    assert cal.source == path


def test_placeholder_when_file_missing(tmp_path):
    cal = SharpCalibration.load_or_placeholder(str(tmp_path / "missing.json"))
    assert cal.source == "placeholder"
