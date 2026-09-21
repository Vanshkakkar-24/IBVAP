"""Robust multi-object tracker for continuous person tracking with motion prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.detection.person_detector import RawPersonDetection
from app.schemas.face import BoundingBox, FaceDetection
from app.tracking.tracker import bbox_iou


@dataclass
class PersonTrackState:
    """State of an actively tracked person."""

    track_id: int
    bbox: BoundingBox
    confidence: float = 1.0
    hits: int = 1
    missed: int = 0
    # Velocity: dx, dy, dw, dh for motion smoothing and coasting
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    face: FaceDetection | None = None
    global_person_id: str | None = None
    reid_embedding: list[float] | None = None

    @property
    def has_face(self) -> bool:
        return self.face is not None

    def predict_next_bbox(self) -> BoundingBox:
        """Estimate next frame position based on current velocity."""
        vx, vy, vw, vh = self.velocity
        new_x = max(0, int(round(self.bbox.x + vx)))
        new_y = max(0, int(round(self.bbox.y + vy)))
        new_w = max(10, int(round(self.bbox.width + vw)))
        new_h = max(10, int(round(self.bbox.height + vh)))
        return BoundingBox(x=new_x, y=new_y, width=new_w, height=new_h)

    def update_measurement(self, new_bbox: BoundingBox, confidence: float) -> None:
        """Update track with a newly matched detection, updating velocity."""
        alpha = 0.35  # Exponential smoothing for velocity
        dx = new_bbox.x - self.bbox.x
        dy = new_bbox.y - self.bbox.y
        dw = new_bbox.width - self.bbox.width
        dh = new_bbox.height - self.bbox.height

        vx, vy, vw, vh = self.velocity
        self.velocity = (
            alpha * dx + (1 - alpha) * vx,
            alpha * dy + (1 - alpha) * vy,
            alpha * dw + (1 - alpha) * vw,
            alpha * dh + (1 - alpha) * vh,
        )

        self.bbox = new_bbox
        self.confidence = confidence
        self.hits += 1
        self.missed = 0


class PersonTracker:
    """Online multi-person tracker using motion-aware IoU association.

    Continuously tracks persons regardless of whether their faces are visible,
    occluded, or turned away.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_missed: int = 30,
        min_hits_to_confirm: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits_to_confirm = min_hits_to_confirm
        self._next_track_id = 1
        self.tracks: list[PersonTrackState] = []

    def update(self, detections: list[RawPersonDetection]) -> list[PersonTrackState]:
        """Update tracker with new frame person detections.

        Returns all currently active and confirmed person tracks.
        """
        # Step 1: Predict positions for existing tracks
        predicted_boxes = {track.track_id: track.predict_next_bbox() for track in self.tracks}

        # Step 2: Compute IoU cost matrix between detections and tracks
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = set(track.track_id for track in self.tracks)
        matched_pairs: list[tuple[int, int]] = []  # (track_id, detection_idx)

        # Greedy bipartite matching
        candidates: list[tuple[float, int, int]] = []
        for d_idx, det in enumerate(detections):
            det_box = BoundingBox(x=det.bbox[0], y=det.bbox[1], width=det.bbox[2], height=det.bbox[3])
            for track in self.tracks:
                pred_box = predicted_boxes[track.track_id]
                # Combine IoU with current box and predicted box
                score_pred = bbox_iou(det_box, pred_box)
                score_curr = bbox_iou(det_box, track.bbox)
                best_score = max(score_pred, score_curr)
                if best_score >= self.iou_threshold:
                    candidates.append((best_score, track.track_id, d_idx))

        # Sort descending by IoU
        candidates.sort(key=lambda item: item[0], reverse=True)
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        for score, t_id, d_idx in candidates:
            if t_id not in assigned_tracks and d_idx not in assigned_dets:
                assigned_tracks.add(t_id)
                assigned_dets.add(d_idx)
                matched_pairs.append((t_id, d_idx))

        # Update matched tracks
        track_map = {t.track_id: t for t in self.tracks}
        for t_id, d_idx in matched_pairs:
            raw = detections[d_idx]
            bbox = BoundingBox(x=raw.bbox[0], y=raw.bbox[1], width=raw.bbox[2], height=raw.bbox[3])
            track_map[t_id].update_measurement(bbox, raw.confidence)

        # Increment missed frames for unmatched tracks
        for track in self.tracks:
            if track.track_id not in assigned_tracks:
                track.missed += 1
                # Coast track using predicted velocity
                if track.missed <= 5:
                    track.bbox = predicted_boxes[track.track_id]

        # Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in assigned_dets:
                raw = detections[d_idx]
                bbox = BoundingBox(x=raw.bbox[0], y=raw.bbox[1], width=raw.bbox[2], height=raw.bbox[3])
                new_track = PersonTrackState(
                    track_id=self._next_track_id,
                    bbox=bbox,
                    confidence=raw.confidence,
                    hits=1,
                    missed=0,
                )
                self._next_track_id += 1
                self.tracks.append(new_track)

        # Cull tracks that exceeded max_missed
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]

        # Return tracks that meet the minimum hits criteria and were not recently lost
        active = [track for track in self.tracks if track.hits >= self.min_hits_to_confirm and track.missed == 0]
        return active

    def reset(self) -> None:
        """Clear all active tracks and reset ID counter."""
        self._next_track_id = 1
        self.tracks.clear()
