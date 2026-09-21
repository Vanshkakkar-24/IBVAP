"""Camera topology configuration and transition validator."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class CameraTopology:
    """Manages spatial relationships and transit-time constraints between cameras."""

    def __init__(self, topology_config: dict[str, dict[str, Any]] | str | None = None) -> None:
        self.rules: dict[str, dict[str, dict[str, Any]]] = {}
        if topology_config:
            self.load(topology_config)

    def load(self, config: dict[str, Any] | str) -> None:
        if isinstance(config, str):
            try:
                parsed = json.loads(config)
            except Exception as exc:
                logger.warning("Failed to parse camera topology JSON: %s", exc)
                parsed = {}
        else:
            parsed = config or {}

        self.rules = parsed
        logger.info("Loaded camera topology with %d source cameras", len(self.rules))

    def add_edge(
        self,
        from_camera: str,
        to_camera: str,
        min_seconds: float = 2.0,
        max_seconds: float = 120.0,
        expected_direction: str | None = None,
    ) -> None:
        if from_camera not in self.rules:
            self.rules[from_camera] = {}
        self.rules[from_camera][to_camera] = {
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
            "expected_direction": expected_direction,
        }

    def evaluate_transition(
        self,
        from_camera: str,
        to_camera: str,
        delta_seconds: float,
        direction: str | None = None,
    ) -> tuple[bool, float, str]:
        """Validate if a camera-to-camera transition is physically plausible.

        Returns (is_valid, modifier_weight, reason).
        modifier_weight is in [0.0, 1.2]:
        - 0.0: Transition impossible (rejected)
        - < 1.0: Penalized
        - 1.0: Normal / unconstrained
        - > 1.0: Boosted (matched topology expectations)
        """
        if from_camera == to_camera:
            return True, 1.0, "Same camera track continuity"

        source_rules = self.rules.get(from_camera, {})
        rule = source_rules.get(to_camera)

        if rule:
            min_sec = float(rule.get("min_seconds", 0.0))
            max_sec = float(rule.get("max_seconds", 3600.0))
            exp_dir = rule.get("expected_direction")

            # Check minimum time constraint
            if delta_seconds < min_sec:
                return False, 0.0, f"Transition too fast ({delta_seconds:.1f}s < min {min_sec:.1f}s)"

            # Check maximum time constraint
            if delta_seconds > max_sec:
                decay = math.exp(-(delta_seconds - max_sec) / 60.0)
                if decay < 0.2:
                    return False, 0.0, f"Transition expired ({delta_seconds:.1f}s > max {max_sec:.1f}s)"
                return True, max(0.2, decay), f"Transition delayed ({delta_seconds:.1f}s)"

            # Score boost for matching topology window
            weight = 1.05
            if exp_dir and direction and exp_dir.upper() in direction.upper():
                weight += 0.1

            return True, min(1.2, weight), f"Valid topological transition ({delta_seconds:.1f}s)"

        # Fallback for cameras without explicit connection: apply time decay
        # Persons don't typically jump cameras instantaneously; require at least 1s
        if delta_seconds < 1.0:
            return False, 0.0, f"Simultaneous detection across distinct cameras ({delta_seconds:.1f}s)"

        # Smooth exponential decay over 5 minutes
        time_factor = math.exp(-delta_seconds / 300.0)
        return True, max(0.4, time_factor), f"Unconstrained transition (decay {time_factor:.2f})"
