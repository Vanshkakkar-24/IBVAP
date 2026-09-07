"""Frame annotation helpers."""

from __future__ import annotations

import cv2
import numpy as np

from app.schemas.face import FaceDetection, FaceDetectionFrame


class FaceDrawer:
    """Draws face metadata on frames for demos and annotated outputs."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def draw(self, frame: np.ndarray, metadata: FaceDetectionFrame) -> np.ndarray:
        if not self.enabled:
            return frame
        output = frame.copy()
        for face in metadata.faces:
            self._draw_face(output, face)
        self._draw_global_info(output, metadata)
        return output

    def _draw_face(self, frame: np.ndarray, face: FaceDetection) -> None:
        x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
        color = (0, 220, 120)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        lines = [
            f"Face: {face.face_id}",
            f"Conf: {face.confidence:.0%}",
            f"Quality: {face.quality.score:.0%}",
        ]
        if face.track_id is not None:
            lines.append(f"Track: {face.track_id}")

        top = max(20, y - 8 - 18 * len(lines))
        for i, text in enumerate(lines):
            y_pos = top + i * 18
            cv2.putText(frame, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _draw_global_info(self, frame: np.ndarray, metadata: FaceDetectionFrame) -> None:
        info = (
            f"Camera: {metadata.camera_id} | Faces: {len(metadata.faces)}"
            f" | FPS: {(metadata.processing_fps or 0):.1f}"
        )
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 520), 38), (20, 20, 20), -1)
        cv2.putText(frame, info, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

