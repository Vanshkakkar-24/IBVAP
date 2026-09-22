"""
Movement-threshold check --- step 4 of the Module 9 flow:

    ... -> Object/zone filter -> [Movement threshold] -> Night movement event -> ...

The prototype note in the design doc explicitly calls out the failure
mode this file exists to prevent:

    "Test shadows/noise false positives."

A static or barely-jittering blob (a shadow moving with headlights, a
flag, sensor noise on a stationary object) must NOT raise a night
movement event just because a bounding box exists at night. Only
*real, sustained, ground-plane displacement* over a time window should
count as movement.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Point = Tuple[float, float]


def path_displacement(points: List[Point]) -> float:
    """
    Net straight-line displacement between the first and last point in
    the window (not total path length). Using net displacement -
    rather than summed segment length - is what filters out small,
    back-and-forth jitter from noise/shadows, since jitter tends to
    cancel out over a window while genuine walking/driving does not.
    """
    if len(points) < 2:
        return 0.0
    (x0, y0), (x1, y1) = points[0], points[-1]
    return math.hypot(x1 - x0, y1 - y0)


def exceeds_movement_threshold(
    trajectory: List[Point],
    timestamps_s: List[float],
    pixel_threshold: float,
    window_s: float,
) -> bool:
    """
    True if net displacement over the most recent `window_s` seconds of
    the trajectory exceeds `pixel_threshold`.

    `trajectory` and `timestamps_s` must be the same length and time-
    ordered (oldest first). `timestamps_s` are seconds since epoch (or
    any consistent monotonic float), not datetime objects, to keep this
    function dependency-free and trivially unit-testable.
    """
    if len(trajectory) < 2 or len(trajectory) != len(timestamps_s):
        return False

    cutoff = timestamps_s[-1] - window_s
    windowed_points = [
        p for p, t in zip(trajectory, timestamps_s) if t >= cutoff
    ]
    if len(windowed_points) < 2:
        return False

    return path_displacement(windowed_points) >= pixel_threshold
