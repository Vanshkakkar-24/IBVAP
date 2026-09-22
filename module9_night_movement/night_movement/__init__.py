"""
Module 9 --- Night-Time Movement Detection
============================================

Identifies configured movement during night periods by combining
detections/tracks (from Module 2 + Module 5) with a configurable
night-time schedule, then emits a normalized event for the
Event Engine (Module 12).

Public API:

    from night_movement import NightMovementDetector, NightScheduleConfig

See README.md for the full design-to-code mapping.
"""

from .models import TrackUpdate, NightMovementEvent
from .config import NightScheduleConfig, load_config_from_db_row
from .detector import NightMovementDetector
from .low_light import enhance_low_light, is_dark_frame, mean_luminance, prepare_detection_frame

__all__ = [
    "TrackUpdate",
    "NightMovementEvent",
    "NightScheduleConfig",
    "load_config_from_db_row",
    "NightMovementDetector",
    "enhance_low_light",
    "is_dark_frame",
    "mean_luminance",
    "prepare_detection_frame",
]
