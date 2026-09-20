import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from night_movement.low_light import is_dark_frame, prepare_detection_frame


def test_dark_scene_is_enhanced_without_mutating_display_frame():
    frame = np.full((40, 40, 3), 25, dtype=np.uint8)
    original = frame.copy()
    inference, enhanced, luminance = prepare_detection_frame(frame, dark_threshold=72.0)
    assert enhanced is True
    assert luminance == 25.0
    assert inference.mean() > frame.mean()
    assert np.array_equal(frame, original)


def test_bright_scene_skips_enhancement():
    frame = np.full((20, 20, 3), 180, dtype=np.uint8)
    inference, enhanced, _ = prepare_detection_frame(frame)
    assert is_dark_frame(frame) is False
    assert enhanced is False
    assert inference is frame
