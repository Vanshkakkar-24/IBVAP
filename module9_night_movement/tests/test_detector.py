"""
Integration tests for NightMovementDetector.

Two of these map directly onto scenario IDs already defined in the
design doc's Section 32.1 ("Recommended scenarios") so the same
pass/fail language used across every module's testing lines up here
too:

    SCN-001  Normal person movement (daytime)      -> Detection, no alert
    SCN-004  Night restricted-zone entry            -> NIGHT_INTRUSION / CRITICAL
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from night_movement.config import NightScheduleConfig
from night_movement.detector import NightMovementDetector
from night_movement.models import TrackUpdate


def make_config(**overrides) -> NightScheduleConfig:
    base = dict(
        camera_id="CAM-001",
        mode="fixed",
        start_hour=18,
        end_hour=6,
        movement_pixel_threshold=25.0,
        movement_time_window_s=3.0,
        min_track_age_s=1.0,
        cooldown_s=30.0,
    )
    base.update(overrides)
    return NightScheduleConfig(**base)


def feed_walk(detector, track_id, object_type, start_time, start_x, dx_per_step, steps, step_s=0.75, zone_breach_at=None):
    """Helper: simulate a track walking steadily to the right."""
    last_event = None
    for i in range(steps):
        ts = start_time + timedelta(seconds=i * step_s)
        x = start_x + i * dx_per_step
        bbox = (x, 380, x + 40, 420)  # bottom-center ground point == (x+20, 420)
        update = TrackUpdate(
            camera_id="CAM-001",
            track_id=track_id,
            object_type=object_type,
            timestamp=ts,
            bbox=bbox,
            confidence=0.9,
            zone_breach=(zone_breach_at is not None and i >= zone_breach_at),
            zone_id="ZONE-01" if (zone_breach_at is not None and i >= zone_breach_at) else None,
        )
        result = detector.process(update)
        if result is not None:
            last_event = result
    return last_event


def test_scn001_daytime_normal_movement_no_alert():
    """SCN-001: normal person movement during the day -> detection, no alert."""
    cfg = make_config()
    detector = NightMovementDetector(cfg)
    daytime = datetime(2026, 1, 1, 14, 0, 0)  # 2 PM

    event = feed_walk(detector, track_id=1, object_type="person",
                       start_time=daytime, start_x=100, dx_per_step=15, steps=6)

    assert event is None


def test_scn004_night_restricted_zone_entry_is_critical():
    """SCN-004: night restricted-zone entry -> NIGHT_INTRUSION / CRITICAL."""
    cfg = make_config()
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)  # 11 PM

    event = feed_walk(detector, track_id=2, object_type="person",
                       start_time=night, start_x=100, dx_per_step=15, steps=6,
                       zone_breach_at=3)

    assert event is not None
    assert event.event_type == "NIGHT_INTRUSION"
    assert event.severity == "CRITICAL"
    assert "ZONE_BREACH" in event.reason_codes
    assert "NIGHT" in event.reason_codes
    assert event.zone_id == "ZONE-01"


def test_plain_night_movement_without_zone_is_high_not_critical():
    """Night movement with NO zone breach should be NIGHT_MOVEMENT / HIGH (person)."""
    cfg = make_config()
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)

    event = feed_walk(detector, track_id=3, object_type="person",
                       start_time=night, start_x=100, dx_per_step=15, steps=6)

    assert event is not None
    assert event.event_type == "NIGHT_MOVEMENT"
    assert event.severity == "HIGH"
    assert "ZONE_BREACH" not in event.reason_codes


def test_vehicle_night_movement_defaults_to_medium_severity():
    cfg = make_config()
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)

    event = feed_walk(detector, track_id=4, object_type="vehicle",
                       start_time=night, start_x=100, dx_per_step=40, steps=6)

    assert event is not None
    assert event.severity == "MEDIUM"


def test_shadow_noise_at_night_does_not_alert():
    """Anti-false-positive: a stationary/jittering blob at night must NOT alert."""
    cfg = make_config()
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)

    last_event = None
    jitter_xs = [100, 101, 99, 100, 102, 99, 100]
    for i, x in enumerate(jitter_xs):
        ts = night + timedelta(seconds=i * 0.5)
        update = TrackUpdate(
            camera_id="CAM-001", track_id=5, object_type="person",
            timestamp=ts, bbox=(x, 380, x + 40, 420), confidence=0.9,
        )
        result = detector.process(update)
        if result is not None:
            last_event = result

    assert last_event is None


def test_unwatched_object_type_is_ignored():
    cfg = make_config(watched_object_types=["person"])
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)

    event = feed_walk(detector, track_id=6, object_type="vehicle",
                       start_time=night, start_x=100, dx_per_step=40, steps=6)

    assert event is None


def test_low_confidence_detection_is_ignored():
    cfg = make_config(min_confidence=0.8)
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)

    update = TrackUpdate(
        camera_id="CAM-001", track_id=7, object_type="person",
        timestamp=night, bbox=(100, 380, 140, 420), confidence=0.4,
    )
    assert detector.process(update) is None


def test_cooldown_prevents_duplicate_events_for_same_track():
    cfg = make_config(cooldown_s=100.0)
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)

    # first burst of movement -> one event
    first = feed_walk(detector, track_id=8, object_type="person",
                       start_time=night, start_x=100, dx_per_step=15, steps=6)
    assert first is not None

    # immediately continue moving -> should be suppressed by cooldown
    second = feed_walk(detector, track_id=8, object_type="person",
                        start_time=night + timedelta(seconds=5), start_x=200, dx_per_step=15, steps=6)
    assert second is None


def test_disabled_config_never_alerts():
    cfg = make_config(enabled=False)
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)

    event = feed_walk(detector, track_id=9, object_type="person",
                       start_time=night, start_x=100, dx_per_step=15, steps=6)
    assert event is None


def test_forget_track_clears_internal_state():
    cfg = make_config()
    detector = NightMovementDetector(cfg)
    night = datetime(2026, 1, 1, 23, 0, 0)
    feed_walk(detector, track_id=10, object_type="person",
              start_time=night, start_x=100, dx_per_step=15, steps=3)

    assert 10 in detector._trajectories
    detector.forget_track(10)
    assert 10 not in detector._trajectories
    assert 10 not in detector._first_seen
