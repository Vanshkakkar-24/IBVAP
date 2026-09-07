"""REST API routes for face detection."""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app.detection.base import ModelLoadError
from app.schemas.face import (
    CommandResponse,
    DetectFrameRequest,
    DetectFrameResponse,
    StartCameraRequest,
    StartRtspRequest,
    StatusResponse,
    UploadResponse,
)
from app.video.webcam import WebcamReader, list_available_cameras

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Render the primary interactive Web Dashboard."""
    template_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse("<h2>Face Detection Hub</h2><p>Template missing.</p>")
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/face/status", response_model=StatusResponse)
async def face_status(request: Request) -> StatusResponse:
    settings = request.app.state.settings
    controller = request.app.state.stream_controller
    return StatusResponse(
        running_source=controller.running_source,
        model_path=str(settings.model_path),
        model_present=Path(settings.model_path).exists(),
        tracking_enabled=settings.enable_tracking,
        visualization_enabled=settings.enable_visualization,
        details={"last_error": controller.last_error},
    )


@router.get("/api/face/metrics")
async def face_metrics(request: Request):
    return request.app.state.face_service.current_metrics()


@router.get("/api/face/cameras")
async def get_cameras() -> dict[str, list[int]]:
    """Probe and return available hardware webcam indices."""
    cams = await asyncio.to_thread(list_available_cameras)
    return {"cameras": cams}


@router.get("/api/face/live-feed")
async def live_feed(
    request: Request,
    camera_index: int = 0,
    conf_threshold: float = 0.65,
    enable_tracking: bool = True,
):
    """High-speed MJPEG stream directly from the system webcam."""
    service = request.app.state.face_service
    websocket_manager = request.app.state.stream_controller.websocket_manager
    loop = asyncio.get_running_loop()

    def frame_generator():
        reader = None
        try:
            reader = WebcamReader(camera_index)
        except Exception as exc:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                f"Camera error: {exc}",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            _, err_jpg = cv2.imencode(".jpg", blank)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + err_jpg.tobytes() + b"\r\n"
            return

        service.tracker.reset()
        try:
            for frame_id, frame in reader.frames(lambda: False):
                metadata = service.process_frame(
                    frame,
                    frame_id=frame_id,
                    camera_id=f"webcam_{camera_index}",
                    input_fps=reader.fps,
                    source_type="webcam",
                )
                annotated = service.draw(frame.copy(), metadata)
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()

                payload = metadata.model_dump(mode="json")
                asyncio.run_coroutine_threadsafe(websocket_manager.broadcast_json(payload), loop)

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                time.sleep(0.01)
        finally:
            if reader:
                reader.release()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/api/face/detect-frame", response_model=DetectFrameResponse)
async def detect_frame(request: Request, payload: DetectFrameRequest) -> DetectFrameResponse:
    """Process a single image frame (e.g. from browser webcam canvas)."""
    service = request.app.state.face_service
    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image encoding: {exc}")

    if frame is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode image frame.")

    metadata = service.process_frame(
        frame,
        frame_id=0,
        camera_id=payload.camera_id or "browser_webcam",
        source_type="browser",
    )

    annotated_b64 = None
    if payload.return_annotated:
        annotated = service.draw(frame.copy(), metadata)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return DetectFrameResponse(
        ok=True,
        metadata=metadata,
        annotated_image_base64=annotated_b64,
    )


@router.post("/api/face/upload", response_model=UploadResponse)
async def upload_video(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    settings = request.app.state.settings
    service = request.app.state.face_service

    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .mp4 uploads are supported.",
        )
    if file.content_type and file.content_type not in {"video/mp4", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}",
        )

    settings.ensure_storage_dirs()
    upload_path = settings.upload_dir / f"{uuid4().hex}.mp4"
    bytes_written = 0
    remove_upload = True
    try:
        with upload_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Upload exceeds {settings.max_upload_mb} MB limit.",
                    )
                output.write(chunk)

        if bytes_written == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        try:
            metadata, metrics, output_video = await asyncio.to_thread(service.process_video_path, upload_path)
        except ModelLoadError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        out_name = output_video.name if output_video else None
        video_url = f"/api/face/video/{out_name}" if out_name else None
        download_url = f"/api/face/download/{out_name}" if out_name else None

        return UploadResponse(
            ok=True,
            message="Video processed successfully.",
            metadata=metadata,
            metrics=metrics,
            output_video=str(output_video) if output_video else None,
            video_url=video_url,
            download_url=download_url,
        )
    finally:
        await file.close()
        if remove_upload and upload_path.exists():
            upload_path.unlink(missing_ok=True)


@router.get("/api/face/video/{filename}")
async def stream_output_video(request: Request, filename: str) -> FileResponse:
    """Stream an annotated output video with browser-compatible headers."""
    settings = request.app.state.settings
    video_path = settings.output_dir / filename
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found.")
    return FileResponse(path=str(video_path), media_type="video/mp4", filename=filename)


@router.get("/api/face/download/{filename}")
async def download_output_video(request: Request, filename: str) -> FileResponse:
    """Download an annotated video file or JSON file."""
    settings = request.app.state.settings
    video_path = settings.output_dir / filename
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return FileResponse(
        path=str(video_path),
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/face/start-camera", response_model=CommandResponse)
async def start_camera(request: Request, payload: StartCameraRequest) -> CommandResponse:
    settings = request.app.state.settings
    controller = request.app.state.stream_controller
    camera_index = payload.camera_index if payload.camera_index is not None else settings.camera_index
    camera_id = payload.camera_id or settings.camera_id
    try:
        controller.start_webcam(camera_index, camera_id, asyncio.get_running_loop())
    except ModelLoadError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return CommandResponse(ok=True, message=f"Started webcam index {camera_index}.")


@router.post("/api/face/stop-camera", response_model=CommandResponse)
async def stop_camera(request: Request) -> CommandResponse:
    stopped = request.app.state.stream_controller.stop()
    return CommandResponse(ok=True, message="Camera stopped." if stopped else "No camera stream was running.")


@router.post("/api/face/start-rtsp", response_model=CommandResponse)
async def start_rtsp(request: Request, payload: StartRtspRequest) -> CommandResponse:
    settings = request.app.state.settings
    controller = request.app.state.stream_controller
    rtsp_url = payload.rtsp_url or settings.rtsp_url
    if not rtsp_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RTSP URL is required. Provide rtsp_url or set RTSP_URL.",
        )
    camera_id = payload.camera_id or settings.camera_id
    try:
        controller.start_rtsp(rtsp_url, camera_id, asyncio.get_running_loop())
    except ModelLoadError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return CommandResponse(ok=True, message="Started RTSP stream.")


@router.post("/api/face/stop-rtsp", response_model=CommandResponse)
async def stop_rtsp(request: Request) -> CommandResponse:
    stopped = request.app.state.stream_controller.stop()
    return CommandResponse(ok=True, message="RTSP stream stopped." if stopped else "No RTSP stream was running.")
