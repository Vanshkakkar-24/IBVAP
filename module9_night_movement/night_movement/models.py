"""
Data models for Module 9.

These mirror the platform-wide contracts already defined in the system
design doc:

  - Section 8  (Module 5 - Multi-Object Tracking) -> "Track fields"
  - Section 31.4 (Common Data Contracts -> Event)

Module 9 does not invent a new track schema. It *consumes* track
updates published by Module 5 and *produces* events in the same shape
Module 12 (Event Engine) already expects, so integration is a straight
plug-in rather than a translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


Bbox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point = Tuple[float, float]               # (x, y) ground-position proxy


@dataclass
class TrackUpdate:
    """
    One incoming update for a tracked object, as published by
    Module 5 (Multi-Object Tracking).

    `trajectory` is a short rolling history of ground-position points
    (bottom-center of bbox, per the Module 8 "Initial geometry" rule),
    oldest first. Module 9 does not need the *entire* track history -
    only a rolling window long enough to judge real movement vs
    sensor noise - so callers may trim it before publishing.
    """
    camera_id: str
    track_id: int
    object_type: str                 # "person" | "vehicle" | ...
    timestamp: datetime
    bbox: Bbox
    confidence: float
    trajectory: List[Point] = field(default_factory=list)
    zone_id: Optional[str] = None        # set if Module 8 says this track is inside/crossing a configured zone
    zone_breach: bool = False            # True if Module 8 already raised an intrusion for this track at this timestamp
    scene_is_dark: bool = False          # measured from the live frame; permits dark-room operation outside schedule

    def ground_point(self) -> Point:
        """Bottom-center of the bounding box - same ground-position proxy Module 8 uses."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class NightMovementEvent:
    """
    Output of Module 9, shaped exactly like the platform Event contract
    (Section 31.4) so the Event Engine can consume it without any
    field translation.
    """
    event_type: str            # "NIGHT_MOVEMENT" or "NIGHT_INTRUSION"
    severity: str              # "MEDIUM" | "HIGH" | "CRITICAL"
    camera_id: str
    track_id: int
    object_type: str
    timestamp: datetime
    confidence: float
    reason_codes: List[str]
    displacement_px: float
    zone_id: Optional[str] = None
    evidence_id: Optional[str] = None   # filled in later by Module 14 (Evidence Capture)

    def to_dict(self) -> dict:
        d = {
            "event_type": self.event_type,
            "severity": self.severity,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "object_type": self.object_type,
            "timestamp": self.timestamp.isoformat(),
            "confidence": round(self.confidence, 4),
            "reason_codes": self.reason_codes,
            "displacement_px": round(self.displacement_px, 2),
            "evidence_id": self.evidence_id,
        }
        if self.zone_id:
            d["zone_id"] = self.zone_id
        return d
