"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.websocket import manager as websocket_manager
from app.api.websocket import router as websocket_router
from app.cache.redis_client import get_redis_manager
from app.camera.camera_manager import CameraManager
from app.config import get_settings
from app.db.database import init_db
from app.logging_config import configure_logging
from app.reid.global_identity_manager import GlobalIdentityManager
from app.reid.topology import CameraTopology
from app.services.face_service import FaceDetectionService
from app.services.multi_camera_manager import MultiCameraManager
from app.services.stream_runner import StreamController
from app.storage.object_storage import get_storage_service

configure_logging()
settings = get_settings()
settings.ensure_storage_dirs()

# Initialize DB tables
init_db()

topology = CameraTopology(settings.camera_topology_json)
global_identity_manager = GlobalIdentityManager(
    similarity_threshold=settings.reid_similarity_threshold,
    min_observations=settings.min_reid_observations,
    topology=topology,
)

camera_manager = CameraManager(
    settings=settings,
    global_identity_manager=global_identity_manager,
    websocket_manager=websocket_manager,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize registered cameras
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        camera_manager.initialize_from_db(loop)
    except Exception as exc:
        pass
    yield
    # Shutdown
    app.state.stream_controller.stop()
    if hasattr(app.state, "camera_manager"):
        app.state.camera_manager.stop_all()
    if hasattr(app.state, "multi_camera_manager"):
        app.state.multi_camera_manager.stop_all()


app = FastAPI(
    title="IBVAP Multi-Camera System",
    description="Multi-Camera Person Detection, Tracking, Appearance Re-ID, and Face Snapshot System.",
    version="3.0.0",
    lifespan=lifespan,
)

if settings.allowed_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

app.state.settings = settings
app.state.global_identity_manager = global_identity_manager
app.state.camera_manager = camera_manager
app.state.storage_service = get_storage_service()
app.state.redis_manager = get_redis_manager()

app.state.face_service = FaceDetectionService(settings)
app.state.stream_controller = StreamController(app.state.face_service, websocket_manager)
app.state.multi_camera_manager = MultiCameraManager(
    settings,
    websocket_manager,
    global_registry=app.state.face_service.global_registry,
)

app.include_router(api_router)
app.include_router(websocket_router)

