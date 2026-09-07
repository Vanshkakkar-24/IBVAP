from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.video.sampling import FrameSampler
from app.video.video_file import VideoFileReader


def test_mp4_opens_and_frames_can_be_read(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((64, 64, 3), 100, dtype=np.uint8))
    writer.release()

    reader = VideoFileReader(path)
    frames = list(reader.frames())
    assert len(frames) == 3


def test_frame_sampler_can_skip_frames() -> None:
    sampler = FrameSampler(input_fps=30, process_fps=10)
    processed = [frame_id for frame_id in range(10) if sampler.should_process(frame_id)]
    assert processed == [0, 3, 6, 9]

