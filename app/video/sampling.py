"""Frame sampling utilities."""

from __future__ import annotations


class FrameSampler:
    """Decides whether a frame should be processed for a target FPS."""

    def __init__(self, input_fps: float, process_fps: float) -> None:
        self.input_fps = input_fps if input_fps and input_fps > 0 else process_fps
        self.process_fps = process_fps
        self._interval = 1
        if process_fps and process_fps > 0 and self.input_fps and self.input_fps > process_fps:
            self._interval = max(1, round(self.input_fps / process_fps))

    def should_process(self, frame_id: int) -> bool:
        if self.process_fps <= 0:
            return True
        return frame_id % self._interval == 0


