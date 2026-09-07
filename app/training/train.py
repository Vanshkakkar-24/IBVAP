"""Training routine for TinyFaceDetector with PyTorch and ONNX export."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure Windows stdout handles unicode characters printed by exporters
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from app.training.dataset import FaceDataset, capture_webcam_dataset, generate_synthetic_dataset
from app.training.model import TinyFaceDetector


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union between two normalized [x, y, w, h] boxes."""
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def train_model(
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    data_dir: Path | str = "dataset/custom",
    checkpoint_path: Path | str = "models/custom_face_detector.pt",
    onnx_path: Path | str = "models/custom_face_detector.onnx",
    capture_webcam: bool = False,
    camera_index: int = 0,
) -> None:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    manifest = data_path / "annotations.json"

    # Step 1: Prepare data
    if capture_webcam:
        samples = capture_webcam_dataset(data_path, num_samples=40, camera_index=camera_index)
    elif manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} existing samples from {manifest}")
    else:
        print(f"No existing dataset at {manifest}. Generating synthetic face samples...")
        samples = generate_synthetic_dataset(data_path, num_samples=120)
        print(f"Generated {len(samples)} synthetic face samples.")

    if not samples:
        raise RuntimeError("No samples available to train.")

    dataset = FaceDataset(samples, image_size=128)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"\nTraining set: {train_size} samples | Validation set: {val_size} samples")

    # Step 2: Initialize Model, Loss, Optimizer
    device = torch.device("cpu")
    model = TinyFaceDetector(input_size=128).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print("\n" + "=" * 55)
    print("  STARTING FACE DETECTOR MODEL TRAINING")
    print("=" * 55)

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels, bboxes, landmarks in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            landmarks = landmarks.to(device)

            optimizer.zero_grad()
            pred_logits, pred_bboxes, pred_landmarks = model(images)

            loss_cls = bce_loss(pred_logits, labels)

            # Mask bbox and landmark loss for positive face samples only
            pos_mask = (labels > 0.5).squeeze(-1)
            if pos_mask.sum() > 0:
                loss_bbox = smooth_l1(pred_bboxes[pos_mask], bboxes[pos_mask])
                loss_lm = smooth_l1(pred_landmarks[pos_mask], landmarks[pos_mask])
            else:
                loss_bbox = torch.tensor(0.0, device=device)
                loss_lm = torch.tensor(0.0, device=device)

            total_loss = loss_cls + 2.0 * loss_bbox + 1.0 * loss_lm
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * len(images)

        train_loss /= train_size

        # Validation loop
        model.eval()
        val_loss = 0.0
        ious = []
        with torch.no_grad():
            for images, labels, bboxes, landmarks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)
                landmarks = landmarks.to(device)

                pred_logits, pred_bboxes, pred_landmarks = model(images)
                loss_cls = bce_loss(pred_logits, labels)

                pos_mask = (labels > 0.5).squeeze(-1)
                if pos_mask.sum() > 0:
                    loss_bbox = smooth_l1(pred_bboxes[pos_mask], bboxes[pos_mask])
                    loss_lm = smooth_l1(pred_landmarks[pos_mask], landmarks[pos_mask])

                    # Calculate validation IoUs
                    p_box = pred_bboxes[pos_mask].cpu().numpy()
                    t_box = bboxes[pos_mask].cpu().numpy()
                    for pb, tb in zip(p_box, t_box):
                        ious.append(compute_iou(pb, tb))
                else:
                    loss_bbox = torch.tensor(0.0, device=device)
                    loss_lm = torch.tensor(0.0, device=device)

                v_loss = loss_cls + 2.0 * loss_bbox + 1.0 * loss_lm
                val_loss += v_loss.item() * len(images)

        val_loss /= val_size
        mean_iou = sum(ious) / len(ious) if ious else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Mean IoU: {mean_iou:.3f}"
        )

    elapsed = time.perf_counter() - start_time
    print("=" * 55)
    print(f"Training finished in {elapsed:.2f}s")

    # Step 3: Save PyTorch weights
    ckpt = Path(checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt))
    print(f"Saved PyTorch weights -> {ckpt}")

    # Step 4: Export to ONNX format
    onnx_out = Path(onnx_path)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128, device=device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_out),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits", "bbox", "landmarks"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "bbox": {0: "batch_size"},
                "landmarks": {0: "batch_size"},
            },
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_out),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits", "bbox", "landmarks"],
        )
    print(f"Exported ONNX model  -> {onnx_out}")

    # Step 5: Verify ONNX model with onnxruntime
    session = ort.InferenceSession(str(onnx_out))
    test_arr = np.random.randn(1, 3, 128, 128).astype(np.float32)
    outs = session.run(None, {"input": test_arr})
    print(f"ONNX Model Verification: Success! (Outputs: {[o.shape for o in outs]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom TinyFaceDetector and export to ONNX.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001).")
    parser.add_argument("--data-dir", default="dataset/custom", help="Dataset directory.")
    parser.add_argument("--checkpoint", default="models/custom_face_detector.pt", help="PyTorch checkpoint path.")
    parser.add_argument("--export-onnx", default="models/custom_face_detector.onnx", help="Exported ONNX path.")
    parser.add_argument("--capture-webcam", action="store_true", help="Capture real face samples with webcam first.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index for capture.")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        onnx_path=args.export_onnx,
        capture_webcam=args.capture_webcam,
        camera_index=args.camera_index,
    )


if __name__ == "__main__":
    main()
