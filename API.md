# API Reference

Base URL:

```text
http://localhost:8000
```

## GET /health

Returns process health.

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status":"ok"}
```

## GET /api/face/status

Returns current module state.

```bash
curl http://localhost:8000/api/face/status
```

Fields include model path, model presence, active source, tracking state, visualization state, and last stream error.

## GET /api/face/metrics

Returns FPS, latency, inference time, processed frame count, face count, average confidence, average quality, detector name, and execution mode.

```bash
curl http://localhost:8000/api/face/metrics
```

## POST /api/face/upload

Uploads and processes an MP4 file.

```bash
curl -F "file=@sample.mp4;type=video/mp4" http://localhost:8000/api/face/upload
```

PowerShell:

```powershell
curl.exe -F "file=@sample.mp4;type=video/mp4" http://localhost:8000/api/face/upload
```

Errors:

- `400`: empty or unreadable video
- `413`: file larger than `MAX_UPLOAD_MB`
- `415`: unsupported file type
- `503`: model missing or cannot initialize

## POST /api/face/start-camera

Starts webcam processing.

```bash
curl -X POST http://localhost:8000/api/face/start-camera \
  -H "Content-Type: application/json" \
  -d "{\"camera_index\":0,\"camera_id\":\"CAM_01\"}"
```

## POST /api/face/stop-camera

Stops the active webcam stream.

```bash
curl -X POST http://localhost:8000/api/face/stop-camera
```

## POST /api/face/start-rtsp

Starts RTSP processing.

```bash
curl -X POST http://localhost:8000/api/face/start-rtsp \
  -H "Content-Type: application/json" \
  -d "{\"rtsp_url\":\"rtsp://user:password@camera-ip/stream\",\"camera_id\":\"CAM_RTSP_01\"}"
```

Prefer `.env` for RTSP URLs so credentials are not left in shell history.

## POST /api/face/stop-rtsp

Stops the active RTSP stream.

```bash
curl -X POST http://localhost:8000/api/face/stop-rtsp
```

## WS /ws/face-detection

Receives compact `FaceDetectionFrame` JSON for active webcam/RTSP streams.

Python test client:

```python
import asyncio
import websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/face-detection") as ws:
        while True:
            print(await ws.recv())

asyncio.run(main())
```

Start a camera or RTSP stream in another terminal to receive messages.

