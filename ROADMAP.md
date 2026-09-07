# Development Roadmap

## Phase 1: Image Detection

Implement detector interface, YuNet integration, image CLI, metadata schemas, and annotation.

## Phase 2: Webcam

Add OpenCV webcam ingestion, frame sampling, live visualization, and stop behavior.

## Phase 3: MP4

Add streaming MP4 processing, upload validation, annotated output, and batch metrics.

## Phase 4: Quality Metrics

Compute face size, Laplacian blur, brightness, and normalized heuristic usability score.

## Phase 5: FastAPI

Expose health, status, metrics, upload, camera, and RTSP endpoints.

## Phase 6: WebSocket

Broadcast compact frame metadata to dashboards.

## Phase 7: Tracking

Keep optional IoU tracker isolated. Track IDs remain temporary and are not identities.

## Phase 8: Person-Track Association

Integrate future person detector outputs through `PersonFaceAssociator`.

## Phase 9: Performance Optimization

Benchmark frame sampling, model inference time, output writing, and CPU/GPU execution.

## Phase 10: SCRFD/RetinaFace Benchmarking

Add detector implementations behind `FaceDetector`, then compare speed, precision, recall, false positives, false negatives, small-face performance, and low-light performance on the same datasets. Do not publish numbers without measured runs.

