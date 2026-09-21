"""Face-to-person track association using anatomical and geometric spatial reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.schemas.face import BoundingBox, FaceDetection
from app.tracking.tracker import bbox_iou

if TYPE_CHECKING:
    from app.tracking.person_tracker import PersonTrackState


@dataclass(slots=True)
class PersonTrack:
    """External person track representation."""

    track_id: int
    bbox: BoundingBox


class PersonFaceAssociator:
    """Associates faces to person tracks using geometric and anatomical upper-body constraints."""

    def __init__(self, min_iou: float = 0.05, max_head_ratio: float = 0.55) -> None:
        self.min_iou = min_iou
        self.max_head_ratio = max_head_ratio

    def associate(self, faces: list[FaceDetection], persons: list[Any]) -> dict[str, int]:
        """Map face_id -> person track_id for backward compatibility."""
        associations: dict[str, int] = {}
        for face in faces:
            best_track_id: int | None = None
            best_score = 0.0
            for person in persons:
                p_bbox = person.bbox
                t_id = person.track_id
                score = bbox_iou(face.bbox, p_bbox)
                is_in_head = self._is_face_in_upper_person(face.bbox, p_bbox)

                if is_in_head or score > best_score:
                    if is_in_head or score >= self.min_iou:
                        best_score = max(score, 0.5 if is_in_head else score)
                        best_track_id = t_id

            if best_track_id is not None:
                associations[face.face_id] = best_track_id
        return associations

    def attach_faces_to_persons(
        self,
        persons: list[PersonTrackState],
        faces: list[FaceDetection],
    ) -> list[FaceDetection]:
        """Attach detected faces to person track states.

        Returns any faces that could not be assigned to an existing person track.
        """
        # Clear previous frame face bindings
        for person in persons:
            person.face = None

        if not persons or not faces:
            return list(faces)

        unassigned_faces: list[FaceDetection] = []
        assigned_track_ids: set[int] = set()

        for face in faces:
            best_person = None
            best_score = 0.0

            for person in persons:
                if person.track_id in assigned_track_ids:
                    continue

                p_box = person.bbox
                iou = bbox_iou(face.bbox, p_box)
                in_upper = self._is_face_in_upper_person(face.bbox, p_box)

                if in_upper:
                    score = 1.0 + iou
                elif iou >= self.min_iou:
                    score = iou
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_person = person

            if best_person is not None and best_score > 0.0:
                best_person.face = face
                face.track_id = best_person.track_id
                assigned_track_ids.add(best_person.track_id)
            else:
                unassigned_faces.append(face)

        return unassigned_faces

    def _is_face_in_upper_person(self, face: BoundingBox, person: BoundingBox) -> bool:
        """Check if face bounding box is inside or substantially overlapping the upper half of person."""
        fx1, fy1, fx2, fy2 = face.as_xyxy()
        px1, py1, px2, py2 = person.as_xyxy()

        # Upper region of person
        head_max_y = py1 + person.height * self.max_head_ratio

        # Horizontal overlap check
        h_overlap = max(0, min(fx2, px2) - max(fx1, px1))
        face_w = face.width
        h_ratio = h_overlap / face_w if face_w > 0 else 0

        # Vertical check: face center should be within upper person region
        face_cy = fy1 + face.height / 2.0
        v_in_upper = py1 - 20 <= face_cy <= head_max_y

        return h_ratio >= 0.35 and v_in_upper
