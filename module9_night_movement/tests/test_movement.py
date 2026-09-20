import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from night_movement.movement import path_displacement, exceeds_movement_threshold


def test_path_displacement_straight_line():
    pts = [(0, 0), (10, 0), (30, 0)]
    assert path_displacement(pts) == 30.0


def test_path_displacement_needs_two_points():
    assert path_displacement([(5, 5)]) == 0.0
    assert path_displacement([]) == 0.0


def test_jitter_does_not_exceed_threshold():
    # a shadow/noise blob that wobbles a few pixels back and forth
    # should NOT register as movement
    trajectory = [(100, 100), (102, 101), (99, 99), (101, 100), (100, 100)]
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert exceeds_movement_threshold(trajectory, timestamps, pixel_threshold=25.0, window_s=3.0) is False


def test_real_walking_movement_exceeds_threshold():
    # a person walking steadily across the frame
    trajectory = [(100, 400), (110, 400), (125, 400), (140, 400), (160, 400)]
    timestamps = [0.0, 0.75, 1.5, 2.25, 3.0]
    assert exceeds_movement_threshold(trajectory, timestamps, pixel_threshold=25.0, window_s=3.0) is True


def test_only_recent_window_counts():
    # object moved a lot a while ago, then became still - should
    # NOT count once the movement falls outside the time window
    trajectory = [(0, 0), (200, 0), (200, 0), (200, 0), (200, 0)]
    timestamps = [0.0, 1.0, 5.0, 6.0, 7.0]
    # window_s=3 -> only last 3s (t=4..7) considered -> points at t=5,6,7 -> no movement
    assert exceeds_movement_threshold(trajectory, timestamps, pixel_threshold=25.0, window_s=3.0) is False


def test_mismatched_lengths_returns_false():
    assert exceeds_movement_threshold([(0, 0), (1, 1)], [0.0], 10.0, 3.0) is False


def test_empty_trajectory_returns_false():
    assert exceeds_movement_threshold([], [], 10.0, 3.0) is False
