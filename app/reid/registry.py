"""Centralized multi-camera identity registry for cross-camera person tracking and Re-ID."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.reid.extractor import cosine_similarity
from app.schemas.face import CrossCameraTransition, GlobalPersonInfo

logger = logging.getLogger(__name__)


class GlobalPersonEntry:
    """Registry entry for a globally recognized person."""

    def __init__(
        self,
        global_person_id: str,
        embedding: list[float],
        camera_id: str,
        has_face: bool = False,
    ) -> None:
        self.global_person_id = global_person_id
        self.embedding = np.array(embedding, dtype=np.float32)
        self.current_camera = camera_id
        self.last_seen = datetime.now(timezone.utc)
        self.seen_count = 1
        self.has_face = has_face
        self.camera_history = [camera_id]

    def update(self, embedding: list[float], camera_id: str, has_face: bool, ema_alpha: float = 0.15) -> None:
        """Update gallery embedding with exponential moving average."""
        new_feat = np.array(embedding, dtype=np.float32)
        # Update running mean representation
        self.embedding = (1.0 - ema_alpha) * self.embedding + ema_alpha * new_feat
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding /= norm

        self.last_seen = datetime.now(timezone.utc)
        self.seen_count += 1
        self.has_face = self.has_face or has_face

        if camera_id not in self.camera_history:
            self.camera_history.append(camera_id)
        self.current_camera = camera_id


class GlobalPersonRegistry:
    """Orchestrates cross-camera Re-ID, minting global IDs and logging transitions."""

    def __init__(
        self,
        similarity_threshold: float = 0.60,
        max_gallery_size: int = 150,
        ema_alpha: float = 0.15,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        self.ema_alpha = ema_alpha

        self._next_id = 1
        self.gallery: dict[str, GlobalPersonEntry] = {}
        self.transitions: list[CrossCameraTransition] = []
        # Mapping: (camera_id, local_track_id) -> global_person_id for temporal consistency
        self.local_to_global: dict[tuple[str, int], str] = {}

    def match_or_register(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: list[float],
        has_face: bool = False,
    ) -> tuple[str, bool, float]:
        """Match detection embedding to existing global identity or create a new global person.

        Returns (global_person_id, is_new_person, similarity_score).
        """
        # Check if this local track in this camera is already linked to a global ID
        key = (camera_id, local_track_id)
        if key in self.local_to_global:
            g_id = self.local_to_global[key]
            if g_id in self.gallery:
                entry = self.gallery[g_id]
                entry.update(embedding, camera_id, has_face, self.ema_alpha)
                return g_id, False, 1.0

        # Compare against gallery
        best_g_id: str | None = None
        best_sim = -1.0

        for g_id, entry in self.gallery.items():
            sim = cosine_similarity(embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_g_id = g_id

        # Match threshold test
        if best_g_id is not None and best_sim >= self.similarity_threshold:
            matched_entry = self.gallery[best_g_id]
            prev_camera = matched_entry.current_camera

            if prev_camera != camera_id:
                # Log cross-camera transition!
                transition = CrossCameraTransition(
                    global_person_id=best_g_id,
                    from_camera=prev_camera,
                    to_camera=camera_id,
                    timestamp=datetime.now(timezone.utc),
                    similarity_score=round(best_sim, 4),
                )
                self.transitions.append(transition)
                logger.info(
                    "Cross-Camera Transition: %s moved from %s to %s (sim: %.3f)",
                    best_g_id,
                    prev_camera,
                    camera_id,
                    best_sim,
                )

            matched_entry.update(embedding, camera_id, has_face, self.ema_alpha)
            self.local_to_global[key] = best_g_id
            return best_g_id, False, best_sim

        # Mint new global person
        new_g_id = f"PERSON_{self._next_id:03d}"
        self._next_id += 1

        new_entry = GlobalPersonEntry(
            global_person_id=new_g_id,
            embedding=embedding,
            camera_id=camera_id,
            has_face=has_face,
        )
        self.gallery[new_g_id] = new_entry
        self.local_to_global[key] = new_g_id
        logger.info("Registered new global identity: %s from camera=%s", new_g_id, camera_id)
        return new_g_id, True, 1.0

    def get_active_persons(self) -> list[GlobalPersonInfo]:
        """Return metadata for all registered global persons."""
        return [
            GlobalPersonInfo(
                global_person_id=e.global_person_id,
                current_camera=e.current_camera,
                last_seen=e.last_seen,
                seen_count=e.seen_count,
                has_face=e.has_face,
                camera_history=list(e.camera_history),
            )
            for e in self.gallery.values()
        ]

    def get_transitions(self) -> list[CrossCameraTransition]:
        """Return history of cross-camera handoffs."""
        return list(self.transitions)

    def reset(self) -> None:
        """Clear all registry state."""
        self._next_id = 1
        self.gallery.clear()
        self.transitions.clear()
        self.local_to_global.clear()
