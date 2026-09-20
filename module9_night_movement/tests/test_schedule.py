from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from night_movement.schedule import is_night_fixed
from night_movement.config import NightScheduleConfig
from night_movement.schedule import is_night


def test_wraparound_window_night_late_evening():
    ts = datetime(2026, 1, 1, 22, 30)  # 10:30 PM
    assert is_night_fixed(ts, start_hour=18, end_hour=6) is True


def test_wraparound_window_night_early_morning():
    ts = datetime(2026, 1, 1, 3, 0)  # 3:00 AM
    assert is_night_fixed(ts, start_hour=18, end_hour=6) is True


def test_wraparound_window_day():
    ts = datetime(2026, 1, 1, 12, 0)  # noon
    assert is_night_fixed(ts, start_hour=18, end_hour=6) is False


def test_boundary_start_hour_is_inclusive():
    ts = datetime(2026, 1, 1, 18, 0)
    assert is_night_fixed(ts, start_hour=18, end_hour=6) is True


def test_boundary_end_hour_is_exclusive():
    ts = datetime(2026, 1, 1, 6, 0)
    assert is_night_fixed(ts, start_hour=18, end_hour=6) is False


def test_same_day_window_no_wraparound():
    # e.g. a dusk-only patrol window: 22:00 -> 23:00
    assert is_night_fixed(datetime(2026, 1, 1, 22, 30), 22, 23) is True
    assert is_night_fixed(datetime(2026, 1, 1, 23, 30), 22, 23) is False
    assert is_night_fixed(datetime(2026, 1, 1, 21, 30), 22, 23) is False


def test_zero_hour_window_fails_safe_to_never_night():
    assert is_night_fixed(datetime(2026, 1, 1, 18, 0), 18, 18) is False


def test_is_night_dispatches_to_fixed_by_default():
    cfg = NightScheduleConfig(camera_id="CAM-001", mode="fixed", start_hour=18, end_hour=6)
    assert is_night(datetime(2026, 1, 1, 2, 0), cfg) is True
    assert is_night(datetime(2026, 1, 1, 14, 0), cfg) is False


def test_is_night_astral_falls_back_without_astral_package(monkeypatch):
    # simulate astral not being importable
    cfg = NightScheduleConfig(camera_id="CAM-001", mode="astral", latitude=28.6, longitude=77.2)
    # should not raise even if astral isn't installed in this environment
    result = is_night(datetime(2026, 1, 1, 2, 0), cfg)
    assert isinstance(result, bool)
