# SIH Face Detection Module

Production-structured face detection module for webcam, RTSP/CCTV streams, uploaded MP4 files, and CLI testing. This module detects visible human faces, draws annotations, emits structured metadata, tracks temporary detections when enabled, and exposes REST/WebSocket integration points.

This is not a face recognition module. It does not identify people, create embeddings, match identities, store biometric profiles, or attach names/identity numbers to faces.

## Features

- YuNet face detector behind a `FaceDetector` interface
- Configurable confidence threshold and minimum face size checks
- Heuristic face usability score from size, blur, and brightness
- Optional temporary IoU tracking with non-identity `track_id`
- Webcam, RTSP, image, and MP4 processing
- Annotated image/video output when enabled
- FastAPI endpoints and real-time WebSocket metadata
- Metrics for FPS, latency, inference time, face counts, confidence, and quality
- Tests for detector pipeline, quality, video ingestion, and API validation
- Docker and Docker Compose support

## Architecture

```text
Input
  -> Video ingestion
  -> Frame sampling
  -> YuNet detector
  -> Confidence and size filters
  -> Quality assessment
  -> Optional temporary tracker
  -> Pydantic metadata
  -> Visualization
  -> REST/WebSocket integration
```

The detector is isolated in `app/detection/base.py`, so SCRFD or RetinaFace can be added later by implementing the same `FaceDetector.detect(frame)` contract.

## Project Structure

```text
app/
  api/
  association/
  detection/
  quality/
  schemas/
  services/
  tracking/
  video/
  visualization/
models/
tests/
uploads/
outputs/
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Setup

Download YuNet from OpenCV Zoo and save it as:

```text
models/face_detection_yunet.onnx
```

PowerShell:

```powershell
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models/face_detection_yunet.onnx"
```

curl:

```bash
curl -L -o models/face_detection_yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The OpenCV Zoo YuNet README documents the model family and notes that YuNet is lightweight and designed for face detection. Newer OpenCV Zoo model variants can be tested later by changing `MODEL_PATH`.

## Configuration

```bash
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Important variables:

- `MODEL_PATH`: YuNet ONNX model path
- `CONFIDENCE_THRESHOLD`: minimum detector confidence
- `MIN_FACE_WIDTH`, `MIN_FACE_HEIGHT`: practical size filter
- `PROCESS_FPS`: target processing FPS for sampling
- `CAMERA_INDEX`, `RTSP_URL`, `CAMERA_ID`: input configuration
- `BLUR_THRESHOLD`, `BRIGHTNESS_MIN`, `BRIGHTNESS_MAX`: quality scoring configuration
- `ENABLE_TRACKING`, `ENABLE_VISUALIZATION`: optional pipeline stages
- `DETECTOR_BACKEND`: `opencv` (default YuNet) or `custom` (trained TinyFaceNet)
- `UPLOAD_DIR`, `OUTPUT_DIR`: local storage paths

Detector confidence and face usability are different: confidence is the model's detection score, while quality is a local heuristic based on size, sharpness, and brightness.

## Running

### 1. Interactive Web Dashboard (Recommended)

Start the web server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open your browser to:

```text
http://localhost:8000
```

- **Live Webcam Tab (Primary)**:
  - **Server Hardware Webcam**: High-speed MJPEG video feed streamed directly from your physical camera.
  - **Browser Client Webcam**: Alternative WebRTC camera capture in your browser.
  - Live HUD displaying FPS, face count, detection latency, confidence, and quality scores.
  - Dynamic controls: Camera index switcher, Confidence threshold slider, Tracking toggle, Snapshot button.
- **Upload Video Tab**:
  - Drag-and-drop MP4 video uploader with progress tracking.
  - Instant in-browser playback of annotated video output.
  - Download buttons for annotated MP4 video and JSON metadata.
- **Model & Training Tab**:
  - Model status, architecture metrics, and one-click training guide.

### 2. Live Webcam Detection (CLI)

Scan for connected webcams:

```bash
python -m app.detect_webcam --list-cameras
```

Run live detection on camera index 0:

```bash
python -m app.detect_webcam --camera-index 0
```

Interactive keyboard shortcuts inside the camera window:
- `[q]` or `[ESC]`: Exit webcam stream
- `[s]`: Save full-resolution annotated snapshot to `outputs/`
- `[t]`: Toggle IoU Tracking on/off dynamically
- `[v]`: Toggle visual overlay on/off

### 3. Video File Detection (CLI)

Process an MP4 video with progress reporting:

```bash
python -m app.detect_video --input sample.mp4
```

Process and preview playback in real-time:

```bash
python -m app.detect_video --input sample.mp4 --show
```

### 4. Custom Model Training Pipeline

Train a custom deep-learning face detector from scratch and export to ONNX:

```bash
python -m app.training.train --epochs 5 --batch-size 16 --lr 0.001
```

Or capture 50 real face training samples from your physical webcam first:

```bash
python -m app.training.train --capture-webcam --epochs 5
```

Artifacts generated:
- PyTorch Weights: `models/custom_face_detector.pt`
- Production ONNX Model: `models/custom_face_detector.onnx`

To run detection with the newly trained custom model, set in `.env` or pass:
```text
DETECTOR_BACKEND=custom
```

### 5. Other CLI Tools

Image:

```bash
python -m app.detect_image --input test.jpg --output outputs/test_annotated.jpg
```

RTSP Stream:

```bash
python -m app.detect_rtsp --url "rtsp://user:password@camera-ip/stream"
```

Benchmark:

```bash
python -m app.benchmark --input sample.mp4 --no-output
```

## API

- `GET /health`
- `GET /api/face/status`
- `GET /api/face/metrics`
- `POST /api/face/upload`
- `POST /api/face/start-camera`
- `POST /api/face/stop-camera`
- `POST /api/face/start-rtsp`
- `POST /api/face/stop-rtsp`
- `WS /ws/face-detection`

Open Swagger UI at:

```text
http://localhost:8000/docs
```

Upload example:

```bash
curl -F "file=@sample.mp4;type=video/mp4" http://localhost:8000/api/face/upload
```

## Testing

```bash
pytest
```

Individual groups:

```bash
pytest tests/test_quality.py
pytest tests/test_detector.py
pytest tests/test_video.py
pytest tests/test_api.py
```

Place optional manual media here:

```text
tests/data/images/
tests/data/videos/
```

Recommended samples: normal face image, crowded image, low-light image, and a short MP4.

## Docker

```bash
docker build -t face-detection .
docker compose up --build
```

Docker works well for API, upload processing, and RTSP. Webcam passthrough differs by host OS and may require device mapping on Linux.

## Privacy And Security

- No recognition or identity matching
- No embeddings
- No face crops stored by default
- Temporary `face_id` and `track_id` only
- Upload type and size validation
- `.env` excluded from git
- RTSP credentials masked in logs
- CORS disabled unless explicitly configured

## Future Improvements

- Add `SCRFDDetector` and `RetinaFaceDetector`
- Benchmark detectors on shared videos
- Add PostgreSQL metadata persistence
- Connect to a React dashboard
- Integrate person detector/tracker using `app/association/person.py`
- Add retention cleanup jobs for uploads and outputs

