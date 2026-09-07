"""Future face-to-person track association helpers."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.face import BoundingBox, FaceDetection
from app.tracking.tracker import bbox_iou


@dataclass(slots=True)
class PersonTrack:
    """External person track from a future person detection module."""

    track_id: int
    bbox: BoundingBox


class PersonFaceAssociator:
    """Associates faces to person tracks using explainable geometry only."""

    def __init__(self, min_iou: float = 0.05) -> None:
        self.min_iou = min_iou

    def associate(self, faces: list[FaceDetection], persons: list[PersonTrack]) -> dict[str, int]:
        associations: dict[str, int] = {}
        for face in faces:
            best_person: PersonTrack | None = None
            best_score = 0.0
            for person in persons:
                score = bbox_iou(face.bbox, person.bbox)
                if score > best_score or self._face_inside_person(face.bbox, person.bbox):
                    best_score = score
                    best_person = person
            if best_person and (best_score >= self.min_iou or self._face_inside_person(face.bbox, best_person.bbox)):
                associations[face.face_id] = best_person.track_id
        return associations

    @staticmethod
    def _face_inside_person(face: BoundingBox, person: BoundingBox) -> bool:
        fx1, fy1, fx2, fy2 = face.as_xyxy()
        px1, py1, px2, py2 = person.as_xyxy()
        return fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2

