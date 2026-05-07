"""
Sweep confidence thresholds for a Faster R-CNN checkpoint and find the one
that gives peak macro-averaged F1 on the validation set.

Strategy (same as tune_yolo_f1.py):
  1. Run a single forward pass at CONF_INFER=0.001 to cache every raw
     prediction with its confidence score.  No GPU work in subsequent steps.
  2. Sweep N_STEPS confidence thresholds from CONF_MIN to CONF_MAX by
     re-filtering the in-memory cache and computing macro-F1 each time.
  3. Report the peak F1 confidence and save results to JSON.

Usage:
    python "tools/tune_fasterrcnn_f1.py"
    python "tools/tune_fasterrcnn_f1.py" --steps 500
    python "tools/tune_fasterrcnn_f1.py" --checkpoint path/to/custom.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "validation" / "labels"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "fasterrcnn_epoch_50.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "fasterrcnn" / "tune_f1"

CLASS_NAMES = ["bird", "drone", "unknown"]
N_CLASSES = len(CLASS_NAMES)
IOU_THRESH = 0.5

# Single inference pass at this conf — captures all predictions for the sweep.
CONF_INFER = 0.001

CONF_MIN = 0.55
CONF_MAX = 0.85
N_STEPS = 200

IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]

_INLINE_STATUS_LEN = 0


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _print_inline_status(status: str) -> None:
    global _INLINE_STATUS_LEN
    if sys.stdout.isatty():
        padded = status.ljust(_INLINE_STATUS_LEN)
        print(f"\r{padded}", end="", flush=True)
        _INLINE_STATUS_LEN = len(status)
    else:
        print(status, flush=True)


def _finish_inline_status_line() -> None:
    global _INLINE_STATUS_LEN
    if sys.stdout.isatty():
        print()
    _INLINE_STATUS_LEN = 0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def load_label_file(label_path: Path) -> tuple[list, list]:
    boxes, categories = [], []
    if not label_path.exists():
        return boxes, categories
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            categories.append(cls if cls in (0, 1) else 2)
            boxes.append((x_center, y_center, w, h))
    return boxes, categories


def xywhn_to_xyxy(box, img_width: int, img_height: int) -> list[float]:
    x_center, y_center, w, h = box
    x1 = (x_center - w / 2.0) * img_width
    y1 = (y_center - h / 2.0) * img_height
    x2 = (x_center + w / 2.0) * img_width
    y2 = (y_center + h / 2.0) * img_height
    return [x1, y1, x2, y2]


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions(gt_boxes: list, pred_boxes: list, iou_thresh: float) -> dict:
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return {}
    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(n_pred):
            ious[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])
    assignments = {}
    while True:
        max_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_idx] < iou_thresh:
            break
        gi, pi = max_idx
        assignments[gi] = pi
        ious[gi, :] = -1.0
        ious[:, pi] = -1.0
    return assignments


def find_validation_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[Path], list[Path]]:
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not label_paths:
        raise ValueError(f"No label files found in {labels_dir}")
    image_paths, valid_label_paths = [], []
    for label_path in label_paths:
        found = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{label_path.stem}{ext}"
            if candidate.exists():
                found = candidate
                break
        if found:
            image_paths.append(found)
            valid_label_paths.append(label_path)
    if not image_paths:
        raise ValueError(f"No validation images found in {images_dir}")
    return image_paths, valid_label_paths


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cls_score_weight = checkpoint["model_state_dict"]["roi_heads.box_predictor.cls_score.weight"]
    num_classes = cls_score_weight.shape[0]
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def image_to_tensor(image_path: Path, device: str):
    image = Image.open(image_path).convert("RGB")
    return (
        torch.from_numpy(np.array(image, dtype="uint8"))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .to(device)
    )


# ---------------------------------------------------------------------------
# Caching + sweep
# ---------------------------------------------------------------------------

def cache_predictions(
    model,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    device: str,
    start_time: float,
) -> list[dict]:
    """Run one inference pass at CONF_INFER and store raw predictions."""
    total = len(image_paths)
    cached = []

    for idx, (image_path, label_path) in enumerate(zip(image_paths, valid_label_paths), 1):
        gt_boxes_xywh, gt_labels = load_label_file(label_path)
        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes_xyxy = [xywhn_to_xyxy(b, width, height) for b in gt_boxes_xywh]

        image_tensor = image_to_tensor(image_path, device)
        with torch.no_grad():
            pred = model([image_tensor])[0]

        preds = []
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if score < CONF_INFER:
                continue
            shifted = int(label) - 1  # torchvision: 0=background, shift back
            cls_id = shifted if shifted in (0, 1) else 2
            preds.append((float(score), cls_id, list(box)))

        cached.append({"gt_boxes": gt_boxes_xyxy, "gt_labels": gt_labels, "preds": preds})

        pct = idx / total * 100.0
        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        _print_inline_status(
            f"  Caching {idx}/{total} ({pct:.1f}%) | Elapsed: {h}:{m:02d}:{s:02d}"
        )

    _finish_inline_status_line()
    return cached


def compute_f1_from_cache(cached: list[dict], conf_thresh: float) -> tuple[float, float, float]:
    """Filter cached predictions at conf_thresh and compute macro-F1. No GPU."""
    matrix = np.zeros((3, 3), dtype=int)

    for entry in cached:
        gt_boxes = entry["gt_boxes"]
        gt_labels = entry["gt_labels"]
        preds = [(s, c, b) for s, c, b in entry["preds"] if s >= conf_thresh]
        pred_boxes = [p[2] for p in preds]
        pred_labels = [p[1] for p in preds]
        assignments = match_predictions(gt_boxes, pred_boxes, IOU_THRESH)
        for gi, gt_label in enumerate(gt_labels):
            if gi in assignments:
                matrix[pred_labels[assignments[gi]], gt_label] += 1
            else:
                matrix[2, gt_label] += 1

    total = int(matrix.sum())
    precision_list, recall_list, f1_list = [], [], []
    for c in range(N_CLASSES):
        tp = int(matrix[c, c])
        fp = int(matrix[c, :].sum() - tp)
        fn = int(matrix[:, c].sum() - tp)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float("nan")
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    macro_prec = float(np.nanmean(precision_list))
    macro_rec = float(np.nanmean(recall_list))
    macro_f1 = float(np.nanmean(f1_list))
    return macro_prec, macro_rec, macro_f1


def sweep_conf_thresholds(cached: list[dict], n_steps: int, start_time: float) -> list[dict]:
    """Sweep thresholds from CONF_MIN to CONF_MAX and return all results."""
    thresholds = np.linspace(CONF_MIN, CONF_MAX, n_steps)
    results = []

    for i, conf in enumerate(thresholds, 1):
        prec, rec, f1 = compute_f1_from_cache(cached, float(conf))
        results.append({"conf_thresh": float(conf), "precision": prec, "recall": rec, "f1": f1})

        pct = i / n_steps * 100.0
        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        _print_inline_status(
            f"  Sweep {i}/{n_steps} ({pct:.1f}%)  conf={conf:.4f}  "
            f"P={prec:.4f}  R={rec:.4f}  F1={f1:.4f} | Elapsed: {h}:{m:02d}:{s:02d}"
        )

    _finish_inline_status_line()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find peak-F1 confidence threshold for a Faster R-CNN checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to checkpoint .pt file (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=N_STEPS,
        help=f"Number of confidence thresholds to evaluate (default: {N_STEPS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cpu, cuda, or auto (default: auto)",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 70)
    print("  Faster R-CNN — Peak-F1 Confidence Sweep")
    print("=" * 70)
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Images dir  : {IMAGES_DIR}")
    print(f"  Labels dir  : {LABELS_DIR}")
    print(f"  IoU thresh  : {IOU_THRESH}")
    print(f"  Infer conf  : {CONF_INFER}  (single pass)")
    print(f"  Sweep range : [{CONF_MIN:.2f}, {CONF_MAX:.2f}]  steps={args.steps}")
    print(f"  Device      : {device}")

    print("\nLoading validation pairs...")
    image_paths, valid_label_paths = find_validation_pairs(IMAGES_DIR, LABELS_DIR)
    print(f"  {len(image_paths)} image-label pairs")

    print("\nLoading model...")
    model = build_model(checkpoint_path, device)

    start_time = time.time()
    print("\nCaching predictions (single inference pass)...")
    cached = cache_predictions(model, image_paths, valid_label_paths, device, start_time)

    print("\nSweeping confidence thresholds...")
    sweep_results = sweep_conf_thresholds(cached, args.steps, start_time)

    # Find peak F1
    best = max(sweep_results, key=lambda r: r["f1"] if not np.isnan(r["f1"]) else -1.0)

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    print(f"\n{'=' * 70}")
    print(f"  Peak F1     : {best['f1']:.4f}")
    print(f"  conf_thresh : {best['conf_thresh']:.6f}")
    print(f"  Precision   : {best['precision']:.4f}")
    print(f"  Recall      : {best['recall']:.4f}")
    print(f"  Total time  : {h}:{m:02d}:{s:02d}")
    print(f"{'=' * 70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = OUTPUT_DIR / "best_fasterrcnn_f1.json"
    payload = {
        "checkpoint": str(checkpoint_path),
        "iou_thresh": IOU_THRESH,
        "conf_infer": CONF_INFER,
        "sweep_min": CONF_MIN,
        "sweep_max": CONF_MAX,
        "n_steps": args.steps,
        "best": {
            "conf_thresh": best["conf_thresh"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
        },
        "sweep": sweep_results,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {output_json}")


if __name__ == "__main__":
    main()
