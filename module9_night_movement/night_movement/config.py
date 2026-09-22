"""
Configuration for Module 9.

Per the design doc's technology stack for this module ("PostgreSQL
configuration"), every tunable lives in one config row per camera so an
operator can change night hours / thresholds from the dashboard without
a code change or restart of the AI pipeline.

Two schedule modes are supported:

  - "fixed"  : simple start_hour/end_hour window (default, no external
               dependency - matches the doc's prototype instruction:
               "Use configurable night hours").
  - "astral" : sunrise/sunset based, using latitude/longitude, for
               sites where a fixed clock window is inaccurate across
               seasons. Optional - only active if the `astral` package
               is installed; falls back to "fixed" otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NightScheduleConfig:
    camera_id: str

    # --- schedule ---
    mode: str = "fixed"                 # "fixed" | "astral"
    start_hour: int = 18                # 24h clock, used when mode == "fixed"
    end_hour: int = 6
    latitude: Optional[float] = None    # required when mode == "astral"
    longitude: Optional[float] = None
    timezone: str = "Asia/Kolkata"

    # --- object / zone filter ---
    watched_object_types: List[str] = field(default_factory=lambda: ["person", "vehicle"])
    min_confidence: float = 0.5

    # --- movement threshold (anti false-positive: shadows/noise/wind) ---
    movement_pixel_threshold: float = 25.0     # min ground-point displacement, in pixels
    movement_time_window_s: float = 3.0        # over this many seconds
    min_track_age_s: float = 1.0               # ignore brand-new, possibly-jittery tracks

    # --- alert-flood control ---
    cooldown_s: float = 30.0            # per track_id, before Module 9 emits another event

    # --- live low-light frame preparation (used before AI detection) ---
    low_light_enabled: bool = True
    dark_luminance_threshold: float = 72.0
    clahe_clip_limit: float = 2.5
    gamma: float = 1.35

    # --- severity base per object type when no zone breach is present ---
    base_severity: dict = field(default_factory=lambda: {"person": "HIGH", "vehicle": "MEDIUM"})

    enabled: bool = True

    def severity_for(self, object_type: str) -> str:
        return self.base_severity.get(object_type, "MEDIUM")


# ---------------------------------------------------------------------
# PostgreSQL row <-> config mapping
# ---------------------------------------------------------------------
#
# DDL (see schema.sql for the full statement):
#
#   CREATE TABLE night_schedule_config (
#       camera_id               TEXT PRIMARY KEY,
#       mode                    TEXT NOT NULL DEFAULT 'fixed',
#       start_hour              SMALLINT NOT NULL DEFAULT 18,
#       end_hour                SMALLINT NOT NULL DEFAULT 6,
#       latitude                DOUBLE PRECISION,
#       longitude               DOUBLE PRECISION,
#       timezone                TEXT NOT NULL DEFAULT 'Asia/Kolkata',
#       watched_object_types    TEXT[] NOT NULL DEFAULT ARRAY['person','vehicle'],
#       min_confidence          REAL NOT NULL DEFAULT 0.5,
#       movement_pixel_threshold REAL NOT NULL DEFAULT 25.0,
#       movement_time_window_s  REAL NOT NULL DEFAULT 3.0,
#       min_track_age_s         REAL NOT NULL DEFAULT 1.0,
#       cooldown_s              REAL NOT NULL DEFAULT 30.0,
#       base_severity_person    TEXT NOT NULL DEFAULT 'HIGH',
#       base_severity_vehicle   TEXT NOT NULL DEFAULT 'MEDIUM',
#       enabled                 BOOLEAN NOT NULL DEFAULT TRUE
#   );
#
def load_config_from_db_row(row: dict) -> NightScheduleConfig:
    """Build a NightScheduleConfig from a dict-like DB row (e.g. from
    SQLAlchemy's `.mappings().first()` or `RealDictCursor`)."""
    return NightScheduleConfig(
        camera_id=row["camera_id"],
        mode=row.get("mode", "fixed"),
        start_hour=row.get("start_hour", 18),
        end_hour=row.get("end_hour", 6),
        latitude=row.get("latitude"),
        longitude=row.get("longitude"),
        timezone=row.get("timezone", "Asia/Kolkata"),
        watched_object_types=list(row.get("watched_object_types") or ["person", "vehicle"]),
        min_confidence=row.get("min_confidence", 0.5),
        movement_pixel_threshold=row.get("movement_pixel_threshold", 25.0),
        movement_time_window_s=row.get("movement_time_window_s", 3.0),
        min_track_age_s=row.get("min_track_age_s", 1.0),
        cooldown_s=row.get("cooldown_s", 30.0),
        low_light_enabled=row.get("low_light_enabled", True),
        dark_luminance_threshold=row.get("dark_luminance_threshold", 72.0),
        clahe_clip_limit=row.get("clahe_clip_limit", 2.5),
        gamma=row.get("gamma", 1.35),
        base_severity={
            "person": row.get("base_severity_person", "HIGH"),
            "vehicle": row.get("base_severity_vehicle", "MEDIUM"),
        },
        enabled=row.get("enabled", True),
    )
