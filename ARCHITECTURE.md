# Architecture

## Pipeline

```text
Video Source (Webcam / RTSP / MP4)
  -> Ingestion & Sampling
  -> Person Detector (YOLOv8 ONNX / OpenCV HOG fallback)
  -> Continuous Person Tracker (Motion-aware IoU / Kalman)  *TRACKS WITHOUT FACE*
  -> Face Detector (YuNet / TinyFaceNet)
  -> Face Quality Assessment & Landmarks
  -> Anatomical Face-to-Person Association (Upper-body geometry)
  -> Person Re-ID Feature Extractor (Spatial HSV Stripes + Deep CNN)
  -> Central GlobalPersonRegistry (Cross-Camera Identity & Transition Tracking)
  -> Drawer / Visualization Overlay
  -> FastAPI / WebSockets / MultiCameraManager
```

## Components

`app/config.py` loads environment variables and validates configuration.

`app/video/` owns frame ingestion. `WebcamReader` uses a camera index, `RtspReader` reconnects after stream interruption, and `VideoFileReader` streams MP4 frames without loading the whole file.

`app/video/sampling.py` limits processed frames to approximately `PROCESS_FPS`.

`app/detection/person_detector.py` defines `PersonDetector`, implementing `YoloPersonDetector` (YOLOv8 ONNX via ONNX Runtime / OpenCV DNN) and `HOGPersonDetector` (OpenCV fallback).

`app/detection/yunet.py` implements YuNet through OpenCV `FaceDetectorYN`.

`app/tracking/person_tracker.py` implements `PersonTracker`, maintaining continuous person identities even when the face is occluded, turned away, or invisible.

`app/association/person.py` implements `PersonFaceAssociator`, linking detected faces to person tracks using anatomical upper-body geometry.

`app/reid/extractor.py` extracts 256-D L2-normalized appearance embeddings (spatial HSV stripes + convolutional features).

`app/reid/registry.py` provides `GlobalPersonRegistry`, matching persons across different camera streams using cosine similarity and logging cross-camera handoffs.

`app/services/multi_camera_manager.py` manages multiple concurrent camera streams and synchronizes them with the central `GlobalPersonRegistry`.

`app/schemas/face.py` defines Pydantic metadata contracts for persons, faces, transitions, and camera streams.

`app/visualization/drawer.py` draws person bounding boxes, track IDs, global IDs, face boxes, and `[Face Hidden - Tracking Person]` status badge.

`app/services/face_service.py` coordinates the complete pipeline for images, frames, and MP4 files.

`app/services/stream_runner.py` runs webcam/RTSP processing in a background thread and publishes metadata to WebSocket clients.

## Integration Flow

```text
React Dashboard
  -> POST /api/face/start-camera or /api/face/start-rtsp
  -> connect /ws/face-detection
  -> render compact FaceDetectionFrame JSON
  -> query /api/face/metrics
```

For uploaded videos:

```text
Client
  -> POST /api/face/upload
  -> server stores temporary MP4
  -> frame-by-frame processing
  -> optional annotated output in outputs/
  -> metadata and metrics response
```

## Detector Swap

To add SCRFD:

1. Create `app/detection/scrfd.py`.
2. Implement `FaceDetector.detect(frame) -> list[RawFaceDetection]`.
3. Add a factory in `FaceDetectionService.detector`.
4. Reuse the same postprocess, quality, tracking, metadata, API, and CLI layers.

The same approach applies to RetinaFace.

