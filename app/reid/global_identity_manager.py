"""Global Identity Manager for appearance-based cross-camera Person Re-ID.

CRITICAL: STRICTLY NO FACE RECOGNITION.
All identity matching is based exclusively on whole-body appearance embeddings,
temporal consistency, camera topology, and motion direction.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.cache.redis_client import get_redis_manager
from app.db.database import get_db_session
from app.db.repository import IBVAPRepository
from app.reid.extractor import cosine_similarity
from app.reid.topology import CameraTopology

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GlobalPersonProfile:
    """In-memory profile of a recognized global person identity."""

    def __init__(
        self,
        global_person_id: str,
        embedding: list[float],
        camera_id: str,
        first_seen: datetime | None = None,
    ) -> None:
        self.global_person_id = global_person_id
        self.embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding /= norm

        now = first_seen or utc_now()
        self.first_seen = now
        self.last_seen = now
        self.current_camera = camera_id
        self.seen_count = 1
        self.camera_history: list[str] = [camera_id]
        self.has_face: bool = False

    def update(
        self,
        new_embedding: list[float],
        camera_id: str,
        has_face: bool = False,
        ema_alpha: float = 0.15,
    ) -> None:
        """Update representative embedding with Exponential Moving Average (EMA)."""
        new_feat = np.array(new_embedding, dtype=np.float32)
        new_norm = np.linalg.norm(new_feat)
        if new_norm > 1e-6:
            new_feat /= new_norm

        self.embedding = (1.0 - ema_alpha) * self.embedding + ema_alpha * new_feat
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding /= norm

        self.last_seen = utc_now()
        self.seen_count += 1
        self.has_face = self.has_face or has_face

        if camera_id not in self.camera_history:
            self.camera_history.append(camera_id)
        self.current_camera = camera_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_person_id": self.global_person_id,
            "current_camera": self.current_camera,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "seen_count": self.seen_count,
            "has_face": self.has_face,
            "camera_history": list(self.camera_history),
        }


class GlobalIdentityManager:
    """Coordinates cross-camera association using body Re-ID, topology, and temporal gating."""

    def __init__(
        self,
        similarity_threshold: float = 0.60,
        ema_alpha: float = 0.15,
        min_observations: int = 2,
        topology: CameraTopology | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.min_observations = min_observations
        self.topology = topology or CameraTopology()

        self._lock = threading.Lock()
        self._next_id = 1
        self.gallery: dict[str, GlobalPersonProfile] = {}
        # Mapping: (camera_id, local_track_id) -> global_person_id
        self.local_to_global: dict[tuple[str, int], str] = {}
        # Track observation counters: (camera_id, local_track_id) -> hit count
        self.track_hits: dict[tuple[str, int], int] = {}
        # Handoff transition logs
        self.transitions: list[dict[str, Any]] = []

    def match_or_register(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: list[float],
        has_face: bool = False,
        movement_direction: str | None = None,
        confidence: float = 1.0,
    ) -> tuple[str, bool, float]:
        """Match body Re-ID embedding against global gallery or register a new identity.

        Returns (global_person_id, is_new_person, similarity_score).
        """
        with self._lock:
            key = (camera_id, local_track_id)
            self.track_hits[key] = self.track_hits.get(key, 0) + 1
            current_hits = self.track_hits[key]

            # 1. Existing local-to-global binding check
            if key in self.local_to_global:
                gid = self.local_to_global[key]
                if gid in self.gallery:
                    profile = self.gallery[gid]
                    profile.update(embedding, camera_id, has_face=has_face, ema_alpha=self.ema_alpha)
                    return gid, False, 1.0

            # 2. Multi-Factor Candidate Scoring
            best_gid: str | None = None
            best_final_score = -1.0
            best_raw_sim = 0.0
            now = utc_now()

            for gid, profile in self.gallery.items():
                # Raw cosine similarity between body embeddings
                raw_sim = cosine_similarity(embedding, profile.embedding)
                if raw_sim < self.similarity_threshold * 0.75:
                    continue  # Early reject weak candidates

                # Calculate temporal gap in seconds
                delta_seconds = max(0.0, (now - profile.last_seen).total_seconds())

                # Validate against camera topology
                is_valid, topo_weight, reason = self.topology.evaluate_transition(
                    from_camera=profile.current_camera,
                    to_camera=camera_id,
                    delta_seconds=delta_seconds,
                    direction=movement_direction,
                )

                if not is_valid:
                    continue

                # Final weighted score
                final_score = raw_sim * topo_weight
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_raw_sim = raw_sim
                    best_gid = gid

            # 3. Association match verification
            if best_gid is not None and best_final_score >= self.similarity_threshold:
                profile = self.gallery[best_gid]
                prev_cam = profile.current_camera

                # Detect cross-camera transition
                if prev_cam != camera_id:
                    transition_record = {
                        "global_person_id": best_gid,
                        "from_camera": prev_cam,
                        "to_camera": camera_id,
                        "timestamp": now.isoformat(),
                        "similarity_score": round(best_raw_sim, 4),
                    }
                    self.transitions.append(transition_record)
                    logger.info(
                        "[%s] %s Track=%d GlobalPerson=%s EVENT=CAMERA_TRANSITION from=%s confidence=%.2f",
                        now.strftime("%H:%M:%S"),
                        camera_id,
                        local_track_id,
                        best_gid,
                        prev_cam,
                        best_raw_sim,
                    )
                    self._persist_transition_event(best_gid, prev_cam, camera_id, best_raw_sim)

                profile.update(embedding, camera_id, has_face=has_face, ema_alpha=self.ema_alpha)
                self.local_to_global[key] = best_gid
                self._persist_person(best_gid, embedding, status="active")
                return best_gid, False, best_final_score

            # 4. Require sufficient observation stability before creating a new global identity
            if current_hits < self.min_observations and confidence < 0.85:
                # Return provisional placeholder until track confirms stability
                provisional_id = f"PROV-{camera_id}-{local_track_id}"
                return provisional_id, False, 0.0

            # 5. Mint new persistent global identity (format: PERSON-00017)
            new_gid = f"PERSON-{self._next_id:05d}"
            self._next_id += 1

            new_profile = GlobalPersonProfile(
                global_person_id=new_gid,
                embedding=embedding,
                camera_id=camera_id,
                first_seen=now,
            )
            new_profile.has_face = has_face
            self.gallery[new_gid] = new_profile
            self.local_to_global[key] = new_gid

            logger.info(
                "[%s] %s Track=%d GlobalPerson=%s EVENT=ENTER new_identity=True",
                now.strftime("%H:%M:%S"),
                camera_id,
                local_track_id,
                new_gid,
            )
            self._persist_person(new_gid, embedding, status="active")
            return new_gid, True, 1.0

    def get_active_persons(self) -> list[dict[str, Any]]:
        with self._lock:
            return [p.to_dict() for p in self.gallery.values()]

    def get_transitions(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.transitions)

    def reset(self) -> None:
        with self._lock:
            self._next_id = 1
            self.gallery.clear()
            self.local_to_global.clear()
            self.track_hits.clear()
            self.transitions.clear()

    @staticmethod
    def _persist_person(global_person_id: str, embedding: list[float], status: str = "active") -> None:
        try:
            with get_db_session() as session:
                repo = IBVAPRepository(session)
                repo.create_or_touch_person(global_person_id, embedding, status=status)
        except Exception as exc:
            logger.debug("Database sync skipped for %s: %s", global_person_id, exc)

        try:
            redis_mgr = get_redis_manager()
            redis_mgr.set_global_person_state(
                global_person_id,
                {"global_person_id": global_person_id, "status": status, "updated_at": utc_now().isoformat()},
            )
        except Exception:
            pass

    @staticmethod
    def _persist_transition_event(
        global_person_id: str,
        from_camera: str,
        to_camera: str,
        similarity: float,
    ) -> None:
        now = utc_now()
        meta = {
            "from_camera": from_camera,
            "to_camera": to_camera,
            "similarity": similarity,
        }
        try:
            with get_db_session() as session:
                repo = IBVAPRepository(session)
                repo.log_event(
                    event_type="CAMERA_TRANSITION",
                    severity="info",
                    person_id=global_person_id,
                    camera_id=to_camera,
                    metadata=meta,
                    timestamp=now,
                )
                repo.log_movement(
                    person_id=global_person_id,
                    camera_id=to_camera,
                    track_id=0,
                    event_type="CAMERA_TRANSITION",
                    confidence=similarity,
                    timestamp=now,
                )
        except Exception as exc:
            logger.debug("Transition event DB sync skipped: %s", exc)

        try:
            redis_mgr = get_redis_manager()
            redis_mgr.push_realtime_event(
                {
                    "event_type": "CAMERA_TRANSITION",
                    "severity": "info",
                    "person_id": global_person_id,
                    "camera_id": to_camera,
                    "metadata": meta,
                    "timestamp": now.isoformat(),
                }
            )
        except Exception:
            pass
