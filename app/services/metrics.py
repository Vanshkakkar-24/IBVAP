"""Performance metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.face import MetricsSnapshot


@dataclass
class MetricsAccumulator:
    execution_mode: str = "cpu"
    detector: str = "YuNet"
    input_fps: float = 0.0
    processed_frames: int = 0
    detected_faces: int = 0
    total_inference_ms: float = 0.0
    total_latency_ms: float = 0.0
    total_confidence: float = 0.0
    total_quality_score: float = 0.0
    processing_started_at: float | None = None
    processing_ended_at: float | None = None

    def update(
        self,
        face_count: int,
        inference_ms: float,
        latency_ms: float,
        confidences: list[float],
        qualities: list[float],
    ) -> None:
        self.processed_frames += 1
        self.detected_faces += face_count
        self.total_inference_ms += inference_ms
        self.total_latency_ms += latency_ms
        self.total_confidence += sum(confidences)
        self.total_quality_score += sum(qualities)

    def snapshot(self) -> MetricsSnapshot:
        elapsed = 0.0
        if self.processing_started_at:
            end = self.processing_ended_at or __import__("time").perf_counter()
            elapsed = max(0.0, end - self.processing_started_at)
        face_count = max(1, self.detected_faces)
        frame_count = max(1, self.processed_frames)
        return MetricsSnapshot(
            input_fps=round(self.input_fps, 3),
            processing_fps=round(self.processed_frames / elapsed, 3) if elapsed else 0.0,
            average_inference_ms=round(self.total_inference_ms / frame_count, 3),
            average_latency_ms=round(self.total_latency_ms / frame_count, 3),
            processed_frames=self.processed_frames,
            detected_faces=self.detected_faces,
            average_confidence=round(self.total_confidence / face_count, 4) if self.detected_faces else 0.0,
            average_quality_score=round(self.total_quality_score / face_count, 4) if self.detected_faces else 0.0,
            execution_mode=self.execution_mode,
            detector=self.detector,
        )

    def reset(self) -> None:
        mode = self.execution_mode
        detector = self.detector
        self.__dict__.update(MetricsAccumulator(execution_mode=mode, detector=detector).__dict__)


