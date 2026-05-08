"""
Compute mAP@0.5 for individual YOLO model checkpoints.

For each discovered YOLO model checkpoint in the target set, this script runs
inference at conf=0.001 on the validation images, then computes per-class
Average Precision at IoU 0.5 using 101-point COCO-style interpolation.

  mAP@0.5 = mean of per-class APs (classes: bird, drone, unknown)

The "unknown" class is a remapping of any label index other than 0 (bird) or
1 (drone), consistent with the rest of the project's evaluation methodology.

Target models: yolo8n, yolo9t, yolo10n, yolo11n, yolo12n, yolo26n

Usage:
    python "tools/eval_yolo_map50.py"
    python "tools/eval_yolo_map50.py" --model yolo8n
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "validation" / "labels"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
OUTPUT_DIR = RUNS_DIR / "yolo_map"
AGGREGATE_JSON = OUTPUT_DIR / "best_yolo_map50.json"

CLASS_NAMES = ["bird", "drone", "unknown"]
N_CLASSES = len(CLASS_NAMES)

# Inference at very low confidence to collect all predictions for the PR curve.
# The conf threshold only affects NMS post-processing, not the forward pass cost,
# so this does not meaningfully slow down inference.
CONF_INFER = 0.001
IOU_THRESH = 0.5
IMGSZ = 640

TARGET_MODELS = {"yolo8m"}
IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]

_INLINE_STATUS_LEN = 0


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _print_inline_status(status: str) -> None:
    """Print status on one updating terminal line, clearing leftovers."""
    global _INLINE_STATUS_LEN
    if sys.stdout.isatty():
        padded = status.ljust(_INLINE_STATUS_LEN)
        print(f"\r{padded}", end="", flush=True)
        _INLINE_STATUS_LEN = len(status)
    else:
        print(status, flush=True)


def _finish_inline_status_line() -> None:
    """Move cursor to the next line after inline updates."""
    global _INLINE_STATUS_LEN
    if sys.stdout.isatty():
        print()
    _INLINE_STATUS_LEN = 0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_label_file(label_path: Path) -> tuple[list, list]:
    """Return (boxes_xywhn, class_ids). Classes >1 are remapped to 2 (unknown)."""
    boxes = []
    categories = []
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
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def find_validation_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[Path], list[Path]]:
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not label_paths:
        raise ValueError(f"No label files found in {labels_dir}")

    image_paths = []
    valid_label_paths = []

    for label_path in label_paths:
        image_name = label_path.stem
        found_image = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{image_name}{ext}"
            if candidate.exists():
                found_image = candidate
                break
        if found_image is None:
            continue
        image_paths.append(found_image)
        valid_label_paths.append(label_path)

    if not image_paths:
        raise ValueError(f"No validation images found in {images_dir}")

    return image_paths, valid_label_paths


def discover_models(runs_dir: Path) -> list[tuple[str, Path]]:
    models = []
    for best_path in sorted(runs_dir.glob("*/weights/best.pt")):
        model_name = best_path.parent.parent.name
        models.append((model_name, best_path))
    return models


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point COCO-style interpolated Average Precision."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recalls >= t
        p = float(precisions[mask].max()) if mask.any() else 0.0
        ap += p / 101.0
    return ap


def compute_map50(
    model,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    start_time: float,
) -> tuple[float, list[dict]]:
    """
    Run inference at CONF_INFER on all validation images and compute mAP@0.5.

    Returns:
        map50            : float — mean AP across classes that have GT
        per_class_results: list of dicts — {class, n_gt, ap}
    """
    total_files = len(image_paths)

    # per_class_preds[c] = list of (confidence, is_tp:int)
    per_class_preds: list[list[tuple[float, int]]] = [[] for _ in range(N_CLASSES)]
    per_class_n_gt = [0] * N_CLASSES

    for idx, (image_path, label_path) in enumerate(zip(image_paths, valid_label_paths), 1):
        # ---- Ground truth ----
        gt_boxes_xywh, gt_labels = load_label_file(label_path)
        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes_xyxy = [xywhn_to_xyxy(b, width, height) for b in gt_boxes_xywh]
        for lbl in gt_labels:
            per_class_n_gt[lbl] += 1

        # ---- Inference ----
        result = model.predict(
            source=str(image_path),
            conf=CONF_INFER,
            imgsz=IMGSZ,
            verbose=False,
        )
        result = result[0] if isinstance(result, list) else result

        pred_boxes_xyxy: list[list[float]] = []
        pred_confs: list[float] = []
        pred_classes: list[int] = []

        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                cls_id = int(cls)
                pred_classes.append(cls_id if cls_id in (0, 1) else 2)
                pred_confs.append(float(conf))
                pred_boxes_xyxy.append(list(box))

        # ---- Per-class greedy matching ----
        for c in range(N_CLASSES):
            c_gt_boxes = [gt_boxes_xyxy[i] for i, lbl in enumerate(gt_labels) if lbl == c]

            c_pred_items = sorted(
                [(pred_confs[j], pred_boxes_xyxy[j]) for j in range(len(pred_classes)) if pred_classes[j] == c],
                key=lambda x: x[0],
                reverse=True,
            )

            matched_gt = set()
            for conf_val, pb in c_pred_items:
                best_iou = 0.0
                best_gt_idx = -1
                for gi, gb in enumerate(c_gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_iou >= IOU_THRESH and best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    per_class_preds[c].append((conf_val, 1))  # TP
                else:
                    per_class_preds[c].append((conf_val, 0))  # FP

        # ---- Progress ----
        pct = idx / total_files * 100.0
        elapsed = time.time() - start_time
        t_hour, t_rem = divmod(int(elapsed), 3600)
        t_min, t_sec = divmod(t_rem, 60)
        _print_inline_status(
            f"  {idx}/{total_files} ({pct:.1f}%) | Elapsed: {t_hour}:{t_min:02d}:{t_sec:02d}"
        )

    _finish_inline_status_line()

    # ---- Compute per-class AP ----
    per_class_results = []
    aps = []

    for c in range(N_CLASSES):
        n_gt = per_class_n_gt[c]
        preds = per_class_preds[c]

        if n_gt == 0:
            # No ground-truth boxes for this class — excluded from mean
            per_class_results.append({"class": CLASS_NAMES[c], "n_gt": 0, "ap": None})
            continue

        if not preds:
            # Model produced no predictions for this class at all
            per_class_results.append({"class": CLASS_NAMES[c], "n_gt": n_gt, "ap": 0.0})
            aps.append(0.0)
            continue

        preds.sort(key=lambda x: x[0], reverse=True)
        tp_arr = np.array([p[1] for p in preds], dtype=np.float64)
        fp_arr = 1.0 - tp_arr
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(fp_arr)

        recalls = cum_tp / n_gt
        precisions = cum_tp / (cum_tp + cum_fp)

        ap = compute_ap_101(recalls, precisions)
        aps.append(ap)
        per_class_results.append({"class": CLASS_NAMES[c], "n_gt": n_gt, "ap": round(float(ap), 6)})

    map50 = float(np.mean(aps)) if aps else 0.0
    return map50, per_class_results


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_single_model(
    model_name: str,
    model_path: Path,
    image_paths: list[Path],
    valid_label_paths: list[Path],
) -> dict | None:
    if not model_path.exists():
        print(f"  [SKIP] {model_name}: checkpoint not found at {model_path}")
        return None

    print(f"\n{'=' * 70}")
    print(f"  Evaluating model : {model_name}")
    print(f"  Checkpoint       : {model_path}")
    print(f"{'=' * 70}")

    model = YOLO(str(model_path))
    start_time = time.time()

    map50, per_class_results = compute_map50(
        model=model,
        image_paths=image_paths,
        valid_label_paths=valid_label_paths,
        start_time=start_time,
    )

    elapsed = time.time() - start_time
    t_hour, t_rem = divmod(int(elapsed), 3600)
    t_min, t_sec = divmod(t_rem, 60)

    result = {
        "model": model_name,
        "model_path": str(model_path),
        "map_50": round(map50, 6),
        "per_class": per_class_results,
        "eval_time_seconds": round(elapsed, 1),
    }

    output_json = OUTPUT_DIR / f"map50_{model_name}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  mAP@0.5 : {map50:.4f}")
    for pc in per_class_results:
        ap_str = f"{pc['ap']:.4f}" if pc["ap"] is not None else "N/A (no GT)"
        print(f"    {pc['class']:10s}: AP={ap_str}  n_gt={pc['n_gt']}")
    print(f"  Time    : {t_hour}:{t_min:02d}:{t_sec:02d}")
    print(f"  Saved   : {output_json}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mAP@0.5 for YOLO model checkpoints (3-class: bird, drone, unknown)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate a single model by run folder name (e.g. yolo8n). Omit to evaluate all target models.",
    )
    args = parser.parse_args()

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")
    if not LABELS_DIR.is_dir():
        raise FileNotFoundError(f"Labels dir not found: {LABELS_DIR}")
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(f"Runs dir not found: {RUNS_DIR}")

    print("=" * 70)
    print("  YOLO mAP@0.5 Evaluator (3-class)")
    print("=" * 70)
    print(f"  Images dir     : {IMAGES_DIR}")
    print(f"  Labels dir     : {LABELS_DIR}")
    print(f"  Target models  : {sorted(TARGET_MODELS)}")
    print(f"  Inference conf : {CONF_INFER}  (low to collect full PR curve)")
    print(f"  IoU threshold  : {IOU_THRESH}")

    print("\nLoading validation images and labels...")
    image_paths, valid_label_paths = find_validation_pairs(IMAGES_DIR, LABELS_DIR)
    print(f"  {len(image_paths)} image-label pairs loaded")

    print("\nDiscovering model checkpoints...")
    discovered_models = discover_models(RUNS_DIR)

    if args.model is not None:
        if args.model not in TARGET_MODELS:
            print(f"  [WARN] '{args.model}' is not in the default target set {sorted(TARGET_MODELS)}, evaluating anyway.")
        target_set = {args.model}
    else:
        target_set = TARGET_MODELS

    discovered_models = [(n, p) for n, p in discovered_models if n in target_set]

    if not discovered_models:
        raise RuntimeError(
            f"No matching checkpoints found for {target_set}. "
            "Ensure runs/detect/<model_name>/weights/best.pt exists."
        )

    print(f"  {len(discovered_models)} model(s) to evaluate:")
    for model_name, model_path in discovered_models:
        print(f"    - {model_name}: {model_path}")

    all_results = {}
    for model_name, model_path in discovered_models:
        try:
            result = evaluate_single_model(
                model_name=model_name,
                model_path=model_path,
                image_paths=image_paths,
                valid_label_paths=valid_label_paths,
            )
            if result is not None:
                all_results[model_name] = result
        except Exception as exc:
            _finish_inline_status_line()
            print(f"\n  [WARN] Failed evaluating {model_name}: {exc}")
            continue

    if not all_results:
        raise RuntimeError("No model produced an evaluation result.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "generated_at_epoch": int(time.time()),
        "settings": {
            "iou_thresh": float(IOU_THRESH),
            "conf_infer": float(CONF_INFER),
            "imgsz": int(IMGSZ),
            "images_dir": str(IMAGES_DIR),
            "labels_dir": str(LABELS_DIR),
        },
        "models": all_results,
    }
    with AGGREGATE_JSON.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nAggregate results saved: {AGGREGATE_JSON}")

    ranking = sorted(all_results.items(), key=lambda kv: kv[1]["map_50"], reverse=True)

    print(f"\n{'=' * 70}")
    print("  mAP@0.5 Ranking")
    print(f"{'=' * 70}")
    for rank, (model_name, res) in enumerate(ranking, 1):
        print(f"  {rank}. {model_name:12s}  mAP@0.5 = {res['map_50']:.4f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
