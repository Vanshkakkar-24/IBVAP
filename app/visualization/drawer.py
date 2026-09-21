"""Frame annotation helpers for person tracking, face detection, and multi-camera display."""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas.face import FaceDetection, FaceDetectionFrame, PersonDetection


class FaceDrawer:
    """Draws person tracking, face detection, and multi-camera metadata on frames."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def draw(self, frame: np.ndarray, metadata: FaceDetectionFrame) -> np.ndarray:
        if not self.enabled:
            return frame
        output = frame.copy()

        # Step 1: Draw tracked persons
        for person in metadata.persons:
            self._draw_person(output, person)

        # Step 2: Draw any unassigned faces that were not inside a person box
        assigned_face_ids = {p.face.face_id for p in metadata.persons if p.face is not None}
        for face in metadata.faces:
            if face.face_id not in assigned_face_ids:
                self._draw_face(output, face, color=(0, 220, 120))

        # Step 3: Global telemetry banner
        self._draw_global_info(output, metadata)
        return output

    def _draw_person(self, frame: np.ndarray, person: PersonDetection) -> None:
        px, py, pw, ph = person.bbox.x, person.bbox.y, person.bbox.width, person.bbox.height
        # Person bounding box: vibrant blue / turquoise
        person_color = (245, 150, 40)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), person_color, 2)

        # Top label for person
        gid = person.global_person_id or f"Person_{person.track_id}"
        header_text = f"{gid} (Track #{person.track_id})"

        # Label background
        (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (px, max(0, py - 22)), (px + tw + 10, max(22, py)), person_color, -1)
        cv2.putText(
            frame,
            header_text,
            (px + 5, max(16, py - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # If face is visible, highlight face box inside person
        if person.has_face and person.face is not None:
            self._draw_face(frame, person.face, color=(0, 230, 120))
            face_status_text = f"Face: Detected ({person.face.confidence:.0%})"
            cv2.putText(
                frame,
                face_status_text,
                (px + 5, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 230, 120),
                1,
                cv2.LINE_AA,
            )
        else:
            # Face not visible - clearly display continuous tracking indicator
            face_status_text = "[Face Hidden - Tracking Person]"
            cv2.putText(
                frame,
                face_status_text,
                (px + 5, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 255),  # Amber
                1,
                cv2.LINE_AA,
            )

    def _draw_face(self, frame: np.ndarray, face: FaceDetection, color: tuple[int, int, int] = (0, 220, 120)) -> None:
        fx, fy, fw, fh = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

        # Draw landmarks if present
        if face.landmarks:
            for pt in face.landmarks:
                cv2.circle(frame, (int(round(pt[0])), int(round(pt[1]))), 2, (255, 200, 0), -1)

        lines = [
            f"Face: {face.face_id}",
            f"Conf: {face.confidence:.0%}",
            f"Quality: {face.quality.score:.0%}",
        ]
        top = max(18, fy - 6 - 16 * len(lines))
        for i, text in enumerate(lines):
            y_pos = top + i * 16
            cv2.putText(frame, text, (fx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _draw_global_info(self, frame: np.ndarray, metadata: FaceDetectionFrame) -> None:
        person_count = len(metadata.persons)
        face_count = len(metadata.faces)
        fps = metadata.processing_fps or 0.0

        info = f"Cam: {metadata.camera_id} | Persons: {person_count} | Faces: {face_count} | FPS: {fps:.1f}"
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 560), 40), (20, 20, 20), -1)
        cv2.putText(frame, info, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
