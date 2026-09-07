# Setup Guide

This guide assumes a fresh Windows development machine. Linux/macOS commands are included where practical.

## 1. Prerequisites

Install:

- Python 3.11 or newer
- Git
- VS Code or another editor
- Optional webcam
- Optional Docker Desktop
- Optional FFmpeg for external video inspection/conversion
- Optional NVIDIA GPU, CUDA, and compatible drivers if you later switch to GPU mode

Verify Python:

```bash
python --version
```

Linux/macOS:

```bash
python3 --version
```

## 2. Project Setup

If using a repository:

```bash
git clone <repository-url>
cd Face-detection
```

If no repository exists yet:

```bash
mkdir Face-detection
cd Face-detection
```

## 3. Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Verify activation:

```bash
python -c "import sys; print(sys.prefix)"
```

The printed path should point to the project `venv`.

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. YuNet Model Setup

The first detector is YuNet from OpenCV Zoo.

Expected local path:

```text
models/face_detection_yunet.onnx
```

PowerShell download:

```powershell
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models/face_detection_yunet.onnx"
```

curl download:

```bash
curl -L -o models/face_detection_yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Verify:

```bash
python -c "from pathlib import Path; print(Path('models/face_detection_yunet.onnx').exists())"
```

Expected output:

```text
True
```

Application lookup is controlled by:

```text
MODEL_PATH=models/face_detection_yunet.onnx
```

## 6. Environment Configuration

Windows:

```bash
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Complete example:

```text
MODEL_PATH=models/face_detection_yunet.onnx
CONFIDENCE_THRESHOLD=0.65
NMS_THRESHOLD=0.30
MIN_FACE_WIDTH=30
MIN_FACE_HEIGHT=30
PROCESS_FPS=10
CAMERA_INDEX=0
RTSP_URL=
CAMERA_ID=CAM_01
BLUR_THRESHOLD=100
BRIGHTNESS_MIN=50
BRIGHTNESS_MAX=210
SIZE_SCORE_REFERENCE_AREA=14400
QUALITY_SIZE_WEIGHT=0.40
QUALITY_BLUR_WEIGHT=0.35
QUALITY_BRIGHTNESS_WEIGHT=0.25
ENABLE_TRACKING=true
ENABLE_VISUALIZATION=true
SAVE_ANNOTATED_OUTPUT=true
DETECTOR_BACKEND=opencv
EXECUTION_PROVIDER=cpu
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
MAX_UPLOAD_MB=500
ALLOWED_CORS_ORIGINS=
RTSP_RECONNECT_SECONDS=5
```

Keep secrets such as RTSP credentials in `.env`. Do not commit `.env`.

## 7. Run The Module

Image:

```bash
python -m app.detect_image --input test.jpg --output outputs/test_annotated.jpg
```

MP4:

```bash
python -m app.detect_video --input sample.mp4
```

Webcam:

```bash
python -m app.detect_webcam --camera-index 0
```

RTSP:

```bash
python -m app.detect_rtsp --url "rtsp://user:password@camera-ip/stream"
```

FastAPI development:

```bash
uvicorn app.main:app --reload
```

Production-like FastAPI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Benchmark:

```bash
python -m app.benchmark --input sample.mp4 --no-output
```

## 8. API Testing

Browser:

```text
http://localhost:8000/docs
```

Health:

```bash
curl http://localhost:8000/health
```

Status:

```bash
curl http://localhost:8000/api/face/status
```

Metrics:

```bash
curl http://localhost:8000/api/face/metrics
```

Upload:

```bash
curl -F "file=@sample.mp4;type=video/mp4" http://localhost:8000/api/face/upload
```

PowerShell upload:

```powershell
curl.exe -F "file=@sample.mp4;type=video/mp4" http://localhost:8000/api/face/upload
```

Start webcam:

```bash
curl -X POST http://localhost:8000/api/face/start-camera -H "Content-Type: application/json" -d "{\"camera_index\":0,\"camera_id\":\"CAM_01\"}"
```

Stop webcam:

```bash
curl -X POST http://localhost:8000/api/face/stop-camera
```

Start RTSP:

```bash
curl -X POST http://localhost:8000/api/face/start-rtsp -H "Content-Type: application/json" -d "{\"camera_id\":\"CAM_RTSP_01\"}"
```

The RTSP example above uses `RTSP_URL` from `.env`.

## 9. WebSocket Testing

Install dependencies, start FastAPI, then run:

```python
import asyncio
import websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/face-detection") as ws:
        while True:
            print(await ws.recv())

asyncio.run(main())
```

Start webcam or RTSP in another terminal. Expected output is JSON containing `camera_id`, `frame_id`, and `faces`.

## 10. Webcam Troubleshooting

Camera not opening:

- Try `CAMERA_INDEX=1` or `--camera-index 1`
- Close Zoom, Teams, browser tabs, or other apps using the camera
- Check Windows camera privacy permissions
- Try another USB port

Black screen:

- Check camera cover and lighting
- Try a lower `PROCESS_FPS`
- Verify OpenCV can read the camera with `python -m app.detect_webcam --camera-index 0`

Low FPS:

- Lower `PROCESS_FPS`
- Disable annotated display with `--no-window`
- Use CPU power mode/performance mode

## 11. MP4 Troubleshooting

Unsupported codec or corrupt MP4:

- Re-encode with FFmpeg:

```bash
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p sample.mp4
```

OpenCV cannot open video:

- Confirm the path is correct
- Try `python -m app.detect_video --input sample.mp4 --no-output`
- Keep very large files for batch runs, not live demos

Annotated outputs are written to:

```text
outputs/
```

## 12. RTSP Troubleshooting

Incorrect URL:

- Verify scheme, host, port, and stream path
- Put credentials in `.env`, not source code

Authentication failure:

- Test the URL in VLC first
- Do not paste passwords into logs or screenshots

Network or disconnection:

- The `RtspReader` reconnects after `RTSP_RECONNECT_SECONDS`
- Check firewall, subnet, and camera power

Codec issue:

- Try a camera substream using H.264
- Validate the stream with FFmpeg or VLC

## 13. GPU Support

CPU mode is the default:

```text
EXECUTION_PROVIDER=cpu
```

GPU mode is prepared but depends on OpenCV CUDA support:

```text
EXECUTION_PROVIDER=gpu
```

To use GPU, install a CUDA-capable OpenCV build and compatible NVIDIA driver/CUDA runtime. If GPU initialization fails, switch back to CPU. ONNX Runtime GPU can be added later for detectors implemented directly with ONNX Runtime.

## 14. Docker Setup

```bash
docker build -t face-detection .
docker compose up --build
```

Notes:

- Mount `models/` so the ONNX file is available inside the container
- RTSP usually works if the container can reach the camera network
- Webcam passthrough is easiest on Linux with device mapping and varies on Windows/macOS

## 15. Test Data

Optional manual samples:

```text
tests/data/images/normal.jpg
tests/data/images/crowded.jpg
tests/data/images/low_light.jpg
tests/data/videos/sample.mp4
```

Run:

```bash
pytest
```

## 16. Privacy Boundary

This module answers:

```text
Is there a face?
Where is it?
How confident is the detector?
How large is it?
How usable is it?
Which temporary track does it belong to?
```

It does not answer:

```text
Who is this person?
```

