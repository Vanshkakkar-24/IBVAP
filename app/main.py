"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.websocket import manager as websocket_manager
from app.api.websocket import router as websocket_router
from app.config import get_settings
from app.logging_config import configure_logging
from app.services.face_service import FaceDetectionService
from app.services.stream_runner import StreamController

configure_logging()
settings = get_settings()
settings.ensure_storage_dirs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    app.state.stream_controller.stop()


app = FastAPI(
    title="SIH Face Detection Module",
    description="Face detection, metadata, quality, visualization, and streaming API. No face recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

if settings.allowed_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

app.state.settings = settings
app.state.face_service = FaceDetectionService(settings)
app.state.stream_controller = StreamController(app.state.face_service, websocket_manager)

app.include_router(api_router)
app.include_router(websocket_router)
