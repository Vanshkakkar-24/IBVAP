# Architecture

## Pipeline

```text
Video Source
  -> Ingestion
  -> Frame Sampling
  -> Preprocessing
  -> Face Detector
  -> Confidence Filter
  -> Minimum Size Check
  -> Face Quality Assessment
  -> Optional Temporary Track Association
  -> Metadata
  -> Visualization
  -> FastAPI / WebSocket
```

## Components

`app/config.py` loads environment variables and validates configuration.

`app/video/` owns frame ingestion. `WebcamReader` uses a camera index, `RtspReader` reconnects after stream interruption, and `VideoFileReader` streams MP4 frames without loading the whole file.

`app/video/sampling.py` limits processed frames to approximately `PROCESS_FPS`.

`app/detection/base.py` defines `FaceDetector`. `app/detection/yunet.py` implements YuNet through OpenCV `FaceDetectorYN`. Future SCRFD or RetinaFace implementations should return the same `RawFaceDetection` objects.

`app/detection/postprocess.py` applies confidence filtering, bounding-box clamping, and minimum-size filtering.

`app/quality/` computes blur, brightness, size, and a normalized heuristic usability score. The score is not a biometric quality guarantee.

`app/tracking/` contains optional temporary IoU tracking. Track IDs are processing-local and do not identify people.

`app/association/person.py` prepares future face-to-person association using geometry. The MVP does not require person detection.

`app/schemas/face.py` defines Pydantic metadata contracts used by API and WebSocket payloads.

`app/visualization/drawer.py` draws bounding boxes, face IDs, confidence, quality, track IDs, camera ID, face count, and FPS.

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

