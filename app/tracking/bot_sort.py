"""BoT-SORT Multi-Object Tracker for robust local person tracking within each camera."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.detection.person_detector import RawPersonDetection
from app.schemas.face import BoundingBox, FaceDetection
from app.tracking.tracker import bbox_iou


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KalmanFilterBBox:
    """Kalman filter for tracking bounding boxes in image space.

    State: [cx, cy, w, h, vcx, vcy, vw, vh]
    """

    def __init__(self, bbox: BoundingBox) -> None:
        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        cx = bbox.x + bbox.width / 2.0
        cy = bbox.y + bbox.height / 2.0
        w = float(bbox.width)
        h = float(bbox.height)

        self.mean = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (observing [cx, cy, w, h])
        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        pos_weight = 1.0 / 20.0
        vel_weight = 1.0 / 160.0

        std_pos = [
            2.0 * pos_weight * w,
            2.0 * pos_weight * h,
            pos_weight * w,
            pos_weight * h,
        ]
        std_vel = [
            10.0 * vel_weight * w,
            10.0 * vel_weight * h,
            vel_weight * w,
            vel_weight * h,
        ]

        self.covariance = np.diag(np.square(std_pos + std_vel))
        self.Q = np.diag(np.square(std_pos + std_vel)) * 0.1
        self.R = np.diag(np.square(std_pos)) * 0.1

    def predict(self) -> None:
        self.mean = np.dot(self.F, self.mean)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

    def update(self, bbox: BoundingBox) -> None:
        cx = bbox.x + bbox.width / 2.0
        cy = bbox.y + bbox.height / 2.0
        w = float(bbox.width)
        h = float(bbox.height)
        measurement = np.array([cx, cy, w, h], dtype=np.float64)

        projected_mean = np.dot(self.H, self.mean)
        projected_cov = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R

        innovation = measurement - projected_mean
        kalman_gain = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(projected_cov))

        self.mean += np.dot(kalman_gain, innovation)
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, self.H), self.covariance)

    def current_bbox(self) -> BoundingBox:
        cx, cy, w, h = self.mean[:4]
        w = max(10, float(w))
        h = max(10, float(h))
        x = max(0, int(round(cx - w / 2.0)))
        y = max(0, int(round(cy - h / 2.0)))
        return BoundingBox(x=x, y=y, width=int(round(w)), height=int(round(h)))


class BoTSORTTrack:
    """Individual person track state maintained by BoT-SORT."""

    def __init__(
        self,
        camera_id: str,
        track_id: int,
        bbox: BoundingBox,
        confidence: float,
    ) -> None:
        self.camera_id = camera_id
        self.local_track_id = track_id
        self.track_id = track_id  # Alias for compatibility
        self.bbox = bbox
        self.confidence = confidence

        self.first_seen = utc_now()
        self.last_seen = utc_now()

        self.kalman = KalmanFilterBBox(bbox)
        self.state = "new"  # new, tracked, lost, removed
        self.hits = 1
        self.missed = 0

        self.appearance_embedding: list[float] | None = None
        self.face: FaceDetection | None = None
        self.global_person_id: str | None = None

        # Position history for movement direction calculation
        center = (bbox.x + bbox.width // 2, bbox.y + bbox.height // 2)
        self.history: list[tuple[int, int]] = [center]
        self.movement_direction: str = "STATIONARY"

    @property
    def current_position(self) -> tuple[int, int]:
        return self.bbox.x + self.bbox.width // 2, self.bbox.y + self.bbox.height // 2

    @property
    def has_face(self) -> bool:
        return self.face is not None

    def predict(self) -> BoundingBox:
        self.kalman.predict()
        self.bbox = self.kalman.current_bbox()
        return self.bbox

    def update(self, new_bbox: BoundingBox, confidence: float) -> None:
        self.kalman.update(new_bbox)
        self.bbox = new_bbox
        self.confidence = confidence
        self.hits += 1
        self.missed = 0
        self.last_seen = utc_now()
        self.state = "tracked"

        # Update position history and direction
        curr_pos = (new_bbox.x + new_bbox.width // 2, new_bbox.y + new_bbox.height // 2)
        self.history.append(curr_pos)
        if len(self.history) > 15:
            self.history.pop(0)

        self.movement_direction = self._calculate_direction()

    def mark_missed(self) -> None:
        self.missed += 1
        self.state = "lost"

    def _calculate_direction(self) -> str:
        if len(self.history) < 3:
            return "STATIONARY"

        # Displacement over the last 5-10 frames
        start_x, start_y = self.history[0]
        end_x, end_y = self.history[-1]

        dx = end_x - start_x
        dy = end_y - start_y
        dist = math.hypot(dx, dy)

        if dist < 8:
            return "STATIONARY"

        # Angle in degrees (-180 to 180)
        angle = math.degrees(math.atan2(-dy, dx))  # Negative dy because screen Y is downward

        if -22.5 <= angle < 22.5:
            return "EAST"
        elif 22.5 <= angle < 67.5:
            return "NORTH_EAST"
        elif 67.5 <= angle < 112.5:
            return "NORTH"
        elif 112.5 <= angle < 157.5:
            return "NORTH_WEST"
        elif angle >= 157.5 or angle < -157.5:
            return "WEST"
        elif -157.5 <= angle < -112.5:
            return "SOUTH_WEST"
        elif -112.5 <= angle < -67.5:
            return "SOUTH"
        else:
            return "SOUTH_EAST"

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "local_track_id": self.local_track_id,
            "global_person_id": self.global_person_id,
            "bbox": [self.bbox.x, self.bbox.y, self.bbox.width, self.bbox.height],
            "confidence": round(self.confidence, 4),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "current_position": list(self.current_position),
            "movement_direction": self.movement_direction,
            "has_face": self.has_face,
        }


class BoTSORTTracker:
    """BoT-SORT tracker for a single camera feed.

    MUST be instantiated independently for each camera stream.
    """

    def __init__(
        self,
        camera_id: str,
        high_thresh: float = 0.45,
        low_thresh: float = 0.15,
        match_thresh: float = 0.35,  # IoU threshold for matching
        track_buffer: int = 60,
        min_hits: int = 2,
    ) -> None:
        self.camera_id = camera_id
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits

        self._next_id = 1
        self.tracked_tracks: list[BoTSORTTrack] = []
        self.lost_tracks: list[BoTSORTTrack] = []

    def update(self, detections: list[RawPersonDetection]) -> list[BoTSORTTrack]:
        """Process new frame detections using 2-stage association.

        Returns all currently active and confirmed tracks.
        """
        # 1. Kalman Prediction for all existing tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # 2. Partition detections into high and low confidence
        high_dets: list[tuple[int, RawPersonDetection, BoundingBox]] = []
        low_dets: list[tuple[int, RawPersonDetection, BoundingBox]] = []

        for idx, det in enumerate(detections):
            bbox = BoundingBox(x=det.bbox[0], y=det.bbox[1], width=det.bbox[2], height=det.bbox[3])
            if det.confidence >= self.high_thresh:
                high_dets.append((idx, det, bbox))
            elif det.confidence >= self.low_thresh:
                low_dets.append((idx, det, bbox))

        # 3. Stage 1: Match high-confidence detections with tracked pool
        track_pool = self.tracked_tracks + self.lost_tracks
        matched_tracks_1, unmatched_tracks_1, unmatched_high_dets = self._associate(
            track_pool, high_dets, self.match_thresh
        )

        for track, (det_idx, det, bbox) in matched_tracks_1:
            track.update(bbox, det.confidence)
            if track in self.lost_tracks:
                self.lost_tracks.remove(track)
                self.tracked_tracks.append(track)

        # 4. Stage 2: Match remaining tracked tracks with low-confidence detections
        remaining_tracked = [t for t in unmatched_tracks_1 if t in self.tracked_tracks]
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            remaining_tracked, low_dets, self.match_thresh
        )

        for track, (det_idx, det, bbox) in matched_tracks_2:
            track.update(bbox, det.confidence)

        # 5. Handle unmatched tracks (transition to lost or remove)
        for track in unmatched_tracks_2:
            track.mark_missed()
            if track in self.tracked_tracks:
                self.tracked_tracks.remove(track)
                self.lost_tracks.append(track)

        # Also mark remaining lost tracks that weren't matched in Stage 1
        remaining_lost = [t for t in unmatched_tracks_1 if t in self.lost_tracks]
        for track in remaining_lost:
            track.mark_missed()

        # Remove tracks that exceeded buffer
        self.lost_tracks = [t for t in self.lost_tracks if t.missed <= self.track_buffer]

        # 6. Initialize new tracks from unmatched high-confidence detections
        for det_idx, det, bbox in unmatched_high_dets:
            new_track = BoTSORTTrack(
                camera_id=self.camera_id,
                track_id=self._next_id,
                bbox=bbox,
                confidence=det.confidence,
            )
            self._next_id += 1
            self.tracked_tracks.append(new_track)

        # 7. Return active tracks meeting confirmation criteria
        active = [
            t for t in self.tracked_tracks
            if (t.hits >= self.min_hits or t.confidence >= 0.70) and t.missed == 0
        ]
        return active

    def reset(self) -> None:
        self._next_id = 1
        self.tracked_tracks.clear()
        self.lost_tracks.clear()

    @staticmethod
    def _associate(
        tracks: list[BoTSORTTrack],
        dets: list[tuple[int, RawPersonDetection, BoundingBox]],
        threshold: float,
    ) -> tuple[
        list[tuple[BoTSORTTrack, tuple[int, RawPersonDetection, BoundingBox]]],
        list[BoTSORTTrack],
        list[tuple[int, RawPersonDetection, BoundingBox]],
    ]:
        if not tracks or not dets:
            return [], list(tracks), list(dets)

        # Compute IoU matrix
        candidates: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(tracks):
            for d_idx, (_, _, d_box) in enumerate(dets):
                iou = bbox_iou(track.bbox, d_box)
                if iou >= threshold:
                    candidates.append((iou, t_idx, d_idx))

        # Greedy match by descending IoU
        candidates.sort(key=lambda x: x[0], reverse=True)
        assigned_t: set[int] = set()
        assigned_d: set[int] = set()
        matches = []

        for score, t_idx, d_idx in candidates:
            if t_idx not in assigned_t and d_idx not in assigned_d:
                assigned_t.add(t_idx)
                assigned_d.add(d_idx)
                matches.append((tracks[t_idx], dets[d_idx]))

        unmatched_tracks = [tracks[i] for i in range(len(tracks)) if i not in assigned_t]
        unmatched_dets = [dets[i] for i in range(len(dets)) if i not in assigned_d]

        return matches, unmatched_tracks, unmatched_dets
