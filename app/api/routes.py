"""REST API routes for face detection."""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app.db.database import get_db, get_db_session
from app.db.repository import IBVAPRepository
from app.detection.base import ModelLoadError
from app.schemas.face import (
    CameraCreateRequest,
    CameraResponse,
    CameraStreamConfig,
    CameraUpdateRequest,
    CommandResponse,
    CrossCameraTransition,
    DetectFrameRequest,
    DetectFrameResponse,
    EventResponse,
    FaceSnapshotResponse,
    GlobalPersonInfo,
    PersonHistoryResponse,
    PersonResponse,
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


@router.get("/api/tracking/global-persons", response_model=list[GlobalPersonInfo])
async def get_global_persons(request: Request) -> list[GlobalPersonInfo]:
    """Retrieve all globally tracked persons across all cameras with Re-ID identities."""
    manager = getattr(request.app.state, "multi_camera_manager", None)
    if manager:
        return manager.get_global_persons()
    return request.app.state.face_service.global_registry.get_active_persons()


@router.get("/api/tracking/transitions", response_model=list[CrossCameraTransition])
async def get_camera_transitions(request: Request) -> list[CrossCameraTransition]:
    """Retrieve history of person movements from one camera to another."""
    manager = getattr(request.app.state, "multi_camera_manager", None)
    if manager:
        return manager.get_transitions()
    return request.app.state.face_service.global_registry.get_transitions()


@router.get("/api/cameras/active")
async def get_active_cameras(request: Request) -> list[dict]:
    """List active camera streams in the multi-camera manager."""
    manager = getattr(request.app.state, "multi_camera_manager", None)
    if manager:
        return manager.get_active_cameras()
    return []


@router.post("/api/cameras/add", response_model=CommandResponse)
async def add_camera_stream(request: Request, payload: CameraStreamConfig) -> CommandResponse:
    """Dynamically register a new camera stream to the multi-camera orchestrator."""
    manager = getattr(request.app.state, "multi_camera_manager", None)
    if not manager:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Multi-camera manager not available.")
    try:
        manager.add_camera(
            camera_id=payload.camera_id,
            source_type=payload.source_type,
            source_value=payload.source_url_or_index,
            loop=asyncio.get_running_loop(),
        )
        return CommandResponse(ok=True, message=f"Started camera {payload.camera_id}.")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/api/cameras/remove", response_model=CommandResponse)
async def remove_camera_stream(request: Request, camera_id: str) -> CommandResponse:
    """Stop and remove a camera stream."""
    manager = getattr(request.app.state, "multi_camera_manager", None)
    if not manager:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Multi-camera manager not available.")
    removed = manager.remove_camera(camera_id)
    return CommandResponse(
        ok=removed,
        message=f"Removed camera {camera_id}." if removed else f"Camera {camera_id} was not active.",
    )


# =====================================================================
# Specification-Compliant REST APIs: /cameras, /persons, /events, /face-snapshots
# =====================================================================


@router.get("/cameras", response_model=list[dict[str, Any]])
async def list_cameras(request: Request) -> list[dict[str, Any]]:
    """List all registered cameras and their runtime status."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if cam_mgr:
        return cam_mgr.get_all_cameras()
    return []


@router.post("/cameras", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def register_camera(request: Request, payload: CameraCreateRequest) -> dict[str, Any]:
    """Register a new camera and optionally launch its AI processing worker."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")

    # Ensure event loop is attached to camera manager
    if not cam_mgr.event_loop:
        cam_mgr.event_loop = asyncio.get_running_loop()

    res = cam_mgr.register_camera(
        camera_id=payload.camera_id,
        name=payload.name,
        stream_url=payload.stream_url,
        camera_type=payload.type,
        location=payload.location,
        enabled=payload.enabled,
        start_immediately=payload.start_immediately,
    )
    return res


@router.get("/cameras/{camera_id}", response_model=dict[str, Any])
async def get_camera_info(request: Request, camera_id: str) -> dict[str, Any]:
    """Get single camera configuration and status."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")
    cam = cam_mgr.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found.")
    return cam


@router.put("/cameras/{camera_id}", response_model=dict[str, Any])
async def update_camera_info(request: Request, camera_id: str, payload: CameraUpdateRequest) -> dict[str, Any]:
    """Update camera configuration."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")
    cam = cam_mgr.update_camera(
        camera_id=camera_id,
        name=payload.name,
        stream_url=payload.stream_url,
        camera_type=payload.type,
        location=payload.location,
        enabled=payload.enabled,
    )
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found.")
    return cam


@router.delete("/cameras/{camera_id}", response_model=CommandResponse)
async def delete_camera_info(request: Request, camera_id: str) -> CommandResponse:
    """Delete a camera from registration."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")
    deleted = cam_mgr.delete_camera(camera_id)
    return CommandResponse(ok=deleted, message=f"Camera '{camera_id}' deleted." if deleted else f"Camera '{camera_id}' not found.")


@router.post("/cameras/{camera_id}/start", response_model=CommandResponse)
async def start_camera_worker(request: Request, camera_id: str) -> CommandResponse:
    """Start an individual camera worker."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")
    if not cam_mgr.event_loop:
        cam_mgr.event_loop = asyncio.get_running_loop()
    started = cam_mgr.start_camera(camera_id)
    if not started:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Could not start camera '{camera_id}'.")
    return CommandResponse(ok=True, message=f"Camera '{camera_id}' worker started.")


@router.post("/cameras/{camera_id}/stop", response_model=CommandResponse)
async def stop_camera_worker(request: Request, camera_id: str) -> CommandResponse:
    """Stop an individual camera worker."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")
    stopped = cam_mgr.stop_camera(camera_id)
    return CommandResponse(ok=stopped, message=f"Camera '{camera_id}' stopped." if stopped else f"Camera '{camera_id}' was not running.")


# ==================== Persons & History ====================


@router.get("/persons", response_model=list[dict[str, Any]])
async def list_persons(request: Request, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """List tracked persons across all cameras."""
    try:
        with get_db_session() as session:
            repo = IBVAPRepository(session)
            persons = repo.get_all_persons(limit=limit, status=status)
            if persons:
                return [p.to_dict() for p in persons]
    except Exception as exc:
        logger.debug("DB query failed for persons: %s", exc)

    # Fallback to active in-memory global registry
    gid_mgr = getattr(request.app.state, "global_identity_manager", None)
    if gid_mgr:
        return gid_mgr.get_active_persons()
    return []


@router.get("/persons/{person_id}", response_model=dict[str, Any])
async def get_person_details(request: Request, person_id: str) -> dict[str, Any]:
    """Retrieve details for a single global person."""
    try:
        with get_db_session() as session:
            repo = IBVAPRepository(session)
            p = repo.get_person_by_global_id(person_id)
            if p:
                return p.to_dict()
    except Exception:
        pass

    gid_mgr = getattr(request.app.state, "global_identity_manager", None)
    if gid_mgr and person_id in gid_mgr.gallery:
        return gid_mgr.gallery[person_id].to_dict()

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Person '{person_id}' not found.")


@router.get("/persons/{person_id}/history", response_model=dict[str, Any])
async def get_person_movement_history(request: Request, person_id: str) -> dict[str, Any]:
    """Retrieve full chronological movement log, camera handoffs, and face snapshots for a person."""
    try:
        with get_db_session() as session:
            repo = IBVAPRepository(session)
            history = repo.get_person_history(person_id)
            if history:
                return history
    except Exception as exc:
        logger.warning("Error fetching history for person %s: %s", person_id, exc)

    # In-memory fallback
    gid_mgr = getattr(request.app.state, "global_identity_manager", None)
    if gid_mgr and person_id in gid_mgr.gallery:
        prof = gid_mgr.gallery[person_id]
        return {
            "person": prof.to_dict(),
            "tracks": [],
            "movements": [
                {
                    "person_id": person_id,
                    "camera_id": prof.current_camera,
                    "event_type": "ENTER",
                    "timestamp": prof.first_seen.isoformat(),
                }
            ],
            "snapshots": [],
        }

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"History for person '{person_id}' not found.")


# ==================== Events & Face Snapshots ====================


@router.get("/events", response_model=list[dict[str, Any]])
async def list_events(
    request: Request,
    severity: str | None = None,
    camera_id: str | None = None,
    person_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List system and tracking events."""
    try:
        with get_db_session() as session:
            repo = IBVAPRepository(session)
            events = repo.get_events(limit=limit, severity=severity, camera_id=camera_id, person_id=person_id)
            if events:
                return [e.to_dict() for e in events]
    except Exception:
        pass

    redis_mgr = getattr(request.app.state, "redis_manager", None)
    if redis_mgr:
        return redis_mgr.get_recent_events(limit=limit)
    return []


@router.get("/events/{event_id}", response_model=dict[str, Any])
async def get_event_detail(request: Request, event_id: int) -> dict[str, Any]:
    """Retrieve single event details."""
    try:
        with get_db_session() as session:
            repo = IBVAPRepository(session)
            ev = repo.get_event(event_id)
            if ev:
                return ev.to_dict()
    except Exception:
        pass
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Event #{event_id} not found.")


@router.get("/face-snapshots", response_model=list[dict[str, Any]])
async def list_face_snapshots(
    request: Request,
    person_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List captured face snapshots (captured without face recognition)."""
    try:
        with get_db_session() as session:
            repo = IBVAPRepository(session)
            snaps = repo.get_face_snapshots(limit=limit, person_id=person_id)
            return [s.to_dict() for s in snaps]
    except Exception as exc:
        logger.debug("Failed to query face snapshots from DB: %s", exc)
        return []


@router.get("/api/snapshots/{filename}")
@router.get("/snapshots/{filename}")
@router.get("/outputs/snapshots/{filename}")
async def serve_snapshot_image(request: Request, filename: str) -> FileResponse:
    """Serve locally stored face snapshots with robust multi-directory path lookup."""
    settings = request.app.state.settings
    # Search in multiple potential candidate directories
    candidates = [
        settings.snapshots_dir / filename,
        Path("outputs/snapshots") / filename,
        Path("outputs/snapshots").resolve() / filename,
        Path(__file__).resolve().parent.parent.parent / "outputs" / "snapshots" / filename,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return FileResponse(path=str(cand), media_type="image/jpeg", filename=filename)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Snapshot image '{filename}' not found.")



@router.get("/cameras/{camera_id}/feed")
async def get_camera_live_feed(request: Request, camera_id: str, annotated: bool = True):
    """Real-time multipart MJPEG video stream for a specific camera worker."""
    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")

    worker = cam_mgr.workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera worker '{camera_id}' not found.")

    def stream_frames():
        while not worker.stop_event.is_set():
            jpeg_data = worker.get_latest_jpeg(annotated=annotated)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_data + b"\r\n"
            time.sleep(0.04)  # ~25 FPS

    return StreamingResponse(
        stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/cameras/{camera_id}/snapshot")
async def get_camera_live_snapshot(request: Request, camera_id: str, annotated: bool = True):
    """Return single current JPEG frame from camera worker."""
    from fastapi import Response

    cam_mgr = getattr(request.app.state, "camera_manager", None)
    if not cam_mgr:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera manager unavailable.")

    worker = cam_mgr.workers.get(camera_id)
    if not worker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera worker '{camera_id}' not found.")

    jpeg_data = worker.get_latest_jpeg(annotated=annotated)
    return Response(content=jpeg_data, media_type="image/jpeg")



