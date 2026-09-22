"""
Optional low-light enhancement --- called out in the design doc's
technology stack for Module 9 as:

    "Optional low-light enhancement later"

This does NOT feed the movement-detection math in movement.py (which
works on already-tracked ground points from Module 5 and stays
resolution/brightness independent by design). Its job is upstream: it
can optionally be applied to frames *before* they reach the
detector/tracker (Module 2/3/4) at night, to improve detection recall
in very dark scenes. It is wired in here, rather than in Module 2,
because "when to enhance" (i.e. only during the configured night
window) is exactly the kind of scheduling decision Module 9 already
owns.

Kept intentionally simple (CLAHE) for the prototype; swap for a
learned low-light enhancement model later without changing the public
function signature.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def mean_luminance(frame_bgr: "np.ndarray") -> float:
    """Return average scene brightness (0=black, 255=white)."""
    if not _HAS_CV2 or frame_bgr is None or frame_bgr.size == 0:
        return 255.0
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())


def is_dark_frame(frame_bgr: "np.ndarray", threshold: float = 72.0) -> bool:
    """True when a frame needs low-light treatment before AI inference."""
    return mean_luminance(frame_bgr) < threshold


def enhance_low_light(
    frame_bgr: "np.ndarray", clip_limit: float = 2.5, tile_grid_size: int = 8,
    gamma: float = 1.35, denoise: bool = False,
) -> "np.ndarray":
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on
    the luminance channel of a BGR frame to lift shadow detail without
    blowing out already-bright areas (e.g. a vehicle's headlights).

    Returns the input frame unchanged if OpenCV isn't available, so
    callers can invoke this unconditionally without guarding imports.
    """
    if not _HAS_CV2:
        return frame_bgr

    # Non-local-means denoising is too slow for a live 720p/1080p webcam
    # (it can make the preview look frozen). Keep the real-time default
    # lightweight; callers processing saved footage can opt into a mild
    # median filter explicitly.
    source = cv2.medianBlur(frame_bgr, 3) if denoise else frame_bgr
    lut = np.array([min(255, ((i / 255.0) ** (1.0 / max(gamma, 0.1))) * 255.0)
                    for i in range(256)], dtype=np.uint8)
    source = cv2.LUT(source, lut)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def prepare_detection_frame(
    frame_bgr: "np.ndarray", *, enabled: bool = True, dark_threshold: float = 72.0,
    clip_limit: float = 2.5, gamma: float = 1.35,
) -> tuple["np.ndarray", bool, float]:
    """Return (inference frame, was enhanced, measured luminance).

    The original frame is retained for display/recording, while only the
    inference copy is adjusted. This keeps CCTV evidence visually faithful.
    """
    luminance = mean_luminance(frame_bgr)
    enhanced = bool(enabled and luminance < dark_threshold)
    if not enhanced:
        return frame_bgr, False, luminance
    return enhance_low_light(frame_bgr, clip_limit=clip_limit, gamma=gamma), True, luminance
