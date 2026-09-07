"""Lightweight IoU-based temporary face tracker."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.face import BoundingBox, FaceDetection
from app.tracking.base import FaceTracker


@dataclass
class _Track:
    track_id: int
    bbox: BoundingBox
    missed: int = 0


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = a.area + b.area - inter_area
    return 0.0 if union <= 0 else inter_area / union


class IoUFaceTracker(FaceTracker):
    """Simple online tracker using IoU overlap only.

    Track IDs are temporary processing IDs. They are not identity IDs.
    """

    def __init__(self, iou_threshold: float = 0.30, max_missed: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_track_id = 1
        self._tracks: list[_Track] = []

    def update(self, detections: list[FaceDetection]) -> list[FaceDetection]:
        assigned_tracks: set[int] = set()
        updated: list[FaceDetection] = []

        for detection in detections:
            best_track: _Track | None = None
            best_iou = 0.0
            for track in self._tracks:
                if track.track_id in assigned_tracks:
                    continue
                score = bbox_iou(detection.bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track and best_iou >= self.iou_threshold:
                best_track.bbox = detection.bbox
                best_track.missed = 0
                assigned_tracks.add(best_track.track_id)
                detection.track_id = best_track.track_id
            else:
                track = _Track(track_id=self._next_track_id, bbox=detection.bbox)
                self._next_track_id += 1
                self._tracks.append(track)
                assigned_tracks.add(track.track_id)
                detection.track_id = track.track_id
            updated.append(detection)

        for track in self._tracks:
            if track.track_id not in assigned_tracks:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]
        return updated

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

