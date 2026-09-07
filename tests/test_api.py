from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_status() -> None:
    response = client.get("/api/face/status")
    assert response.status_code == 200
    assert response.json()["service"] == "face-detection"


def test_upload_validation_rejects_non_mp4() -> None:
    response = client.post(
        "/api/face/upload",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_validation_rejects_empty_mp4() -> None:
    response = client.post(
        "/api/face/upload",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400


def test_dashboard_route() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Face Detection Module" in response.text


def test_get_cameras() -> None:
    response = client.get("/api/face/cameras")
    assert response.status_code == 200
    data = response.json()
    assert "cameras" in data
    assert isinstance(data["cameras"], list)


def test_detect_frame_endpoint() -> None:
    import base64
    import cv2
    import numpy as np

    # Create dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")

    response = client.post(
        "/api/face/detect-frame",
        json={"image_base64": b64, "camera_id": "test_cam", "return_annotated": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "metadata" in data
    assert data["metadata"]["camera_id"] == "test_cam"

