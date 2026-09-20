"""
NightMovementDetector --- the Module 9 orchestrator.

Implements the exact flow from the design doc, section 12:

    Detection + timestamp
        v
    Night schedule check
        v
    Object/zone filter
        v
    Movement threshold
        v
    Night movement event
        v
    Risk/severity
        v
    Alert   (handed off to Module 12 - Event Engine, not built here)

Module 9's job stops at "emit a normalized event." Deduplication across
the *whole* platform, persistence, and actually notifying an operator
are Module 12 / Module 13's job (see design doc sections 15-16). This
detector only applies its own light per-track cooldown so one lingering
intruder doesn't flood the Event Engine with an event every frame.
"""

from __future__ import annotations

import time as _time
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from .config import NightScheduleConfig
from .models import NightMovementEvent, TrackUpdate
from .movement import exceeds_movement_threshold
from .schedule import is_night

_SEVERITY_RANK = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class NightMovementDetector:
    """
    Stateful per-camera detector. Create one instance per camera (or
    key an external dict by camera_id if you prefer a single process
    handling many cameras) and feed it TrackUpdate objects as Module 5
    publishes them.
    """

    #: how many trajectory points to retain in the rolling window,
    #: independent of the timestamps window used for the actual
    #: displacement math (keeps memory bounded even at high FPS)
    MAX_TRAJECTORY_POINTS = 50

    def __init__(self, config: NightScheduleConfig):
        self.config = config

        # track_id -> deque[(ground_point, timestamp_s)]
        self._trajectories: Dict[int, Deque[Tuple[Tuple[float, float], float]]] = defaultdict(
            lambda: deque(maxlen=self.MAX_TRAJECTORY_POINTS)
        )
        # track_id -> first-seen wall-clock seconds (for min_track_age_s)
        self._first_seen: Dict[int, float] = {}
        # track_id -> last time *this detector* emitted an event (cooldown)
        self._last_emitted: Dict[int, float] = {}
        # track_id -> severity of the last emitted event, so a fresh
        # escalation (e.g. plain movement -> zone breach) is never
        # swallowed by a cooldown started by a lower-severity event
        self._last_severity: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def process(self, update: TrackUpdate) -> Optional[NightMovementEvent]:
        """
        Feed one track update through the full flow. Returns a
        NightMovementEvent if (and only if) every stage passes;
        otherwise returns None, meaning "nothing to report yet."
        """
        if not self.config.enabled:
            return None

        ts_epoch = update.timestamp.timestamp()

        # --- 1. Night schedule check ------------------------------------------------
        scheduled_night = is_night(update.timestamp, self.config)
        if not (scheduled_night or update.scene_is_dark):
            # still record trajectory so day->night transitions have
            # history to work with once night starts, but do not alert.
            self._record(update, ts_epoch)
            return None

        # --- 2. Object / zone filter --------------------------------------------------
        if update.object_type not in self.config.watched_object_types:
            self._record(update, ts_epoch)
            return None
        if update.confidence < self.config.min_confidence:
            self._record(update, ts_epoch)
            return None

        self._record(update, ts_epoch)

        # ignore very fresh tracks - not enough history to judge real
        # movement yet, and freshest tracks are the noisiest
        first_seen = self._first_seen[update.track_id]
        if (ts_epoch - first_seen) < self.config.min_track_age_s:
            return None

        # --- 3. Movement threshold ------------------------------------------------
        points, timestamps = self._window_for(update.track_id)
        moved = exceeds_movement_threshold(
            trajectory=points,
            timestamps_s=timestamps,
            pixel_threshold=self.config.movement_pixel_threshold,
            window_s=self.config.movement_time_window_s,
        )
        if not moved and not update.zone_breach:
            # a zone breach from Module 8 is treated as sufficient
            # evidence of meaningful movement on its own (crossing a
            # line is inherently "movement"), even if the net-
            # displacement window hasn't fully filled yet.
            return None

        # --- 5. Night movement event + risk/severity (computed before the
        #        cooldown check, because whether cooldown applies depends
        #        on whether this is an escalation over the last event) ---
        displacement = 0.0
        if len(points) >= 2:
            from .movement import path_displacement
            displacement = path_displacement(points)

        reason_codes: List[str] = ["NIGHT"]
        if update.scene_is_dark and not scheduled_night:
            reason_codes.append("DARK_SCENE")
        if moved:
            reason_codes.append("MOVEMENT_THRESHOLD_EXCEEDED")
        if update.zone_breach:
            reason_codes.append("ZONE_BREACH")

        if update.zone_breach:
            event_type = "NIGHT_INTRUSION"
            severity = "CRITICAL"
        else:
            event_type = "NIGHT_MOVEMENT"
            severity = self.config.severity_for(update.object_type)

        # --- 4. Cooldown (alert-flood control) --------------------------------
        # A cooldown suppresses repeat noise at the *same or lower*
        # severity. It must never suppress a genuine escalation (e.g. a
        # loitering person who then crosses into a restricted zone) -
        # that transition is exactly the kind of event an operator most
        # needs to see promptly.
        last_time = self._last_emitted.get(update.track_id)
        last_severity = self._last_severity.get(update.track_id)
        is_escalation = (
            last_severity is not None
            and _SEVERITY_RANK.get(severity, 0) > _SEVERITY_RANK.get(last_severity, 0)
        )
        if last_time is not None and (ts_epoch - last_time) < self.config.cooldown_s and not is_escalation:
            return None

        self._last_emitted[update.track_id] = ts_epoch
        self._last_severity[update.track_id] = severity

        return NightMovementEvent(
            event_type=event_type,
            severity=severity,
            camera_id=update.camera_id,
            track_id=update.track_id,
            object_type=update.object_type,
            timestamp=update.timestamp,
            confidence=update.confidence,
            reason_codes=reason_codes,
            displacement_px=displacement,
            zone_id=update.zone_id,
        )

    def forget_track(self, track_id: int) -> None:
        """Call when Module 5 reports a track as ended, to free memory."""
        self._trajectories.pop(track_id, None)
        self._first_seen.pop(track_id, None)
        self._last_emitted.pop(track_id, None)
        self._last_severity.pop(track_id, None)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _record(self, update: TrackUpdate, ts_epoch: float) -> None:
        if update.track_id not in self._first_seen:
            self._first_seen[update.track_id] = ts_epoch
        self._trajectories[update.track_id].append((update.ground_point(), ts_epoch))

    def _window_for(self, track_id: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        history = self._trajectories.get(track_id, deque())
        points = [p for p, _ in history]
        timestamps = [t for _, t in history]
        return points, timestamps
