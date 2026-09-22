import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from night_movement import NightMovementDetector, NightScheduleConfig, TrackUpdate


def test_dark_room_override_detects_daytime_motion():
    detector = NightMovementDetector(NightScheduleConfig(
        camera_id="CAM", movement_pixel_threshold=10, min_track_age_s=0,
    ))
    daytime = datetime(2026, 1, 1, 13, 0, 0)
    first = TrackUpdate("CAM", 1, "person", daytime, (0, 0, 20, 40), .9, scene_is_dark=True)
    second = TrackUpdate("CAM", 1, "person", daytime + timedelta(seconds=2), (30, 0, 50, 40), .9, scene_is_dark=True)
    assert detector.process(first) is None
    event = detector.process(second)
    assert event is not None
    assert "DARK_SCENE" in event.reason_codes
