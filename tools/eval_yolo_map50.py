"""
Compute mAP@0.5 for individual YOLO model checkpoints on test dataset.

For each discovered YOLO model checkpoint in the target set, this script runs
inference at conf=0.001 on the test images, then computes per-class
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
IMAGES_DIR = PROJECT_ROOT / "dataset" / "test" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "test" / "labels"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
OUTPUT_DIR = RUNS_DIR / "yolo_map"
AGGREGATE_JSON = OUTPUT_DIR / "best_yolo_map50_test.json"
DATASET_SPLIT = IMAGES_DIR.parent.name  # "test" or "validation"

CLASS_NAMES = ["bird", "drone", "unknown"]
N_CLASSES = len(CLASS_NAMES)

# Inference at very low confidence to collect all predictions for the PR curve.
# The conf threshold only affects NMS post-processing, not the forward pass cost,
# so this does not meaningfully slow down inference.
CONF_INFER = 0.001
IOU_THRESH = 0.5
IMGSZ = 640

TARGET_MODELS: set[str] = set()  # empty = evaluate all discovered models
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
    """Discover model checkpoints from runs/detect/<run_name>/weights/best.pt structure."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    models = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            models.append((run_dir.name, best_pt))
    
    if not models:
        raise FileNotFoundError(
            f"No model checkpoints found in {runs_dir}. "
            "Ensure runs/detect/<run_name>/weights/best.pt exists."
        )
    return models


# ---------------------------------------------------------------------------
# AP computation
# ---------------------------------------------------------------------------

IOU_THRESHOLDS_95 = np.linspace(0.5, 0.95, 10)  # [0.50, 0.55, ..., 0.95]


def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point COCO-style interpolated Average Precision."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recalls >= t
        p = float(precisions[mask].max()) if mask.any() else 0.0
        ap += p / 101.0
    return ap


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

def load_or_build_cache(
    model_name: str,
    model_path: Path,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    start_time: float,
    dataset_split: str = "test",
) -> list[dict]:
    """
    Load cached predictions from disk (shared with tune_yolo_f1.py), or run
    inference and save them.

    Cache file: <model_path.parent>/pred_cache_<model_name>_<split>_imgsz<IMGSZ>.json
    Format (same as tune_yolo_f1.py): list of per-image dicts:
        [{"gt_boxes": [[x1,y1,x2,y2],...], "gt_labels": [int,...],
          "preds": [[conf, cls_id, [x1,y1,x2,y2]], ...]}, ...]
    """
    cache_filename = f"pred_cache_{model_name}_{dataset_split}_imgsz{IMGSZ}.json"
    cache_path = model_path.parent / cache_filename
    total_files = len(image_paths)

    # ---- Try loading existing cache ----
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f)
            if isinstance(cache, list):
                print(f"  Loaded prediction cache: {cache_path}")
                return cache
            # Old dict-format cache — discard and regenerate
            print(f"  [WARN] Cache has old format, regenerating: {cache_path}")
            cache_path.unlink(missing_ok=True)
        except json.JSONDecodeError:
            print(f"  [WARN] Corrupt cache detected, regenerating: {cache_path}")
            cache_path.unlink(missing_ok=True)

    # ---- Run inference ----
    print(f"  Running inference (conf={CONF_INFER}, imgsz={IMGSZ}) on {total_files} images...")
    import torch
    model = YOLO(str(model_path))
    cache: list[dict] = []

    for idx, (image_path, label_path) in enumerate(zip(image_paths, valid_label_paths), 1):
        gt_boxes_xywh, gt_labels = load_label_file(label_path)
        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes_xyxy = [xywhn_to_xyxy(b, width, height) for b in gt_boxes_xywh]

        result = model.predict(
            source=str(image_path),
            conf=CONF_INFER,
            imgsz=IMGSZ,
            verbose=False,
        )
        result = result[0] if isinstance(result, list) else result

        preds = []
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                cls_id = int(cls) if int(cls) in (0, 1) else 2
                preds.append([float(conf), cls_id, [float(v) for v in box]])

        cache.append({"gt_boxes": gt_boxes_xyxy, "gt_labels": gt_labels, "preds": preds})

        pct = idx / total_files * 100.0
        elapsed = time.time() - start_time
        t_hour, t_rem = divmod(int(elapsed), 3600)
        t_min, t_sec = divmod(t_rem, 60)
        _print_inline_status(
            f"  {idx}/{total_files} ({pct:.1f}%) | Elapsed: {t_hour}:{t_min:02d}:{t_sec:02d}"
        )

    _finish_inline_status_line()

    # GPU cleanup
    del model
    torch.cuda.empty_cache()

    # ---- Atomic write ----
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp_path.replace(cache_path)
    print(f"  Saved prediction cache : {cache_path}")

    return cache


def compute_map_metrics(
    cache: list[dict],
) -> tuple[float, float, list[dict]]:
    """
    Compute mAP@0.5 and mAP@0.5-0.95 from cached predictions.

    Cache format (same as tune_yolo_f1.py):
        [{"gt_boxes": [...], "gt_labels": [...], "preds": [[conf, cls_id, [x1,y1,x2,y2]], ...]}, ...]

    Returns:
        map50            : float
        map50_95         : float
        per_class_results: list of dicts — {class, n_gt, ap50, ap50_95}
    """
    # per_class_preds[c] = list of (conf, img_idx, pred_box_xyxy)
    per_class_preds: list[list[tuple[float, int, list[float]]]] = [[] for _ in range(N_CLASSES)]
    # per_class_gts[c][img_idx] = list of gt_box_xyxy
    per_class_gts: list[dict[int, list]] = [{} for _ in range(N_CLASSES)]
    per_class_n_gt = [0] * N_CLASSES

    for img_idx, entry in enumerate(cache):
        gt_labels = entry["gt_labels"]
        gt_boxes_xyxy = entry["gt_boxes"]

        for lbl, gb in zip(gt_labels, gt_boxes_xyxy):
            per_class_n_gt[lbl] += 1
            per_class_gts[lbl].setdefault(img_idx, []).append(gb)

        for conf, cls_id, box in entry["preds"]:
            per_class_preds[int(cls_id)].append((float(conf), img_idx, box))

    # ---- Greedy matching + AP at a given IoU threshold ----
    def _ap_at_iou(c: int, iou_t: float) -> float | None:
        n_gt = per_class_n_gt[c]
        preds = per_class_preds[c]
        if n_gt == 0:
            return None
        if not preds:
            return 0.0

        sorted_preds = sorted(preds, key=lambda x: -x[0])
        matched_gts: dict[int, set] = {}
        tp_list = []

        for conf, img_idx, pb in sorted_preds:
            c_gt_boxes = per_class_gts[c].get(img_idx, [])
            img_matched = matched_gts.setdefault(img_idx, set())
            best_iou, best_gi = 0.0, -1
            for gi, gb in enumerate(c_gt_boxes):
                if gi in img_matched:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_t and best_gi >= 0:
                img_matched.add(best_gi)
                tp_list.append(1)
            else:
                tp_list.append(0)

        tp_arr = np.array(tp_list, dtype=np.float64)
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1.0 - tp_arr)
        recalls = cum_tp / n_gt
        precisions = cum_tp / (cum_tp + cum_fp)
        return compute_ap_101(recalls, precisions)

    # ---- Compute per-class AP at all thresholds ----
    # per_class_all_aps[c][t_idx] = AP or None
    per_class_all_aps = [
        [_ap_at_iou(c, float(iou_t)) for iou_t in IOU_THRESHOLDS_95]
        for c in range(N_CLASSES)
    ]

    # ---- Aggregate ----
    per_class_results = []
    for c in range(N_CLASSES):
        aps = per_class_all_aps[c]
        ap50 = aps[0]  # IoU=0.5 is the first threshold
        valid_aps = [a for a in aps if a is not None]
        ap50_95 = float(np.mean(valid_aps)) if valid_aps else None
        per_class_results.append({
            "class": CLASS_NAMES[c],
            "n_gt": per_class_n_gt[c],
            "ap50": round(float(ap50), 6) if ap50 is not None else None,
            "ap50_95": round(float(ap50_95), 6) if ap50_95 is not None else None,
        })

    valid_ap50 = [r["ap50"] for r in per_class_results if r["ap50"] is not None]
    valid_ap50_95 = [r["ap50_95"] for r in per_class_results if r["ap50_95"] is not None]
    map50 = float(np.mean(valid_ap50)) if valid_ap50 else 0.0
    map50_95 = float(np.mean(valid_ap50_95)) if valid_ap50_95 else 0.0

    return map50, map50_95, per_class_results


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

    start_time = time.time()

    cache = load_or_build_cache(
        model_name=model_name,
        model_path=model_path,
        image_paths=image_paths,
        valid_label_paths=valid_label_paths,
        start_time=start_time,
        dataset_split=DATASET_SPLIT,
    )

    map50, map50_95, per_class_results = compute_map_metrics(
        cache=cache,
    )

    elapsed = time.time() - start_time
    t_hour, t_rem = divmod(int(elapsed), 3600)
    t_min, t_sec = divmod(t_rem, 60)

    result = {
        "model": model_name,
        "model_path": str(model_path),
        "map_50": round(map50, 6),
        "map_50_95": round(map50_95, 6),
        "per_class": per_class_results,
        "eval_time_seconds": round(elapsed, 1),
    }

    output_json = OUTPUT_DIR / f"map50_{model_name}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  mAP@0.5     : {map50:.4f}")
    print(f"  mAP@0.5-0.95: {map50_95:.4f}")
    for pc in per_class_results:
        ap50_str = f"{pc['ap50']:.4f}" if pc["ap50"] is not None else "N/A"
        ap95_str = f"{pc['ap50_95']:.4f}" if pc["ap50_95"] is not None else "N/A"
        print(f"    {pc['class']:10s}: AP50={ap50_str}  AP50-95={ap95_str}  n_gt={pc['n_gt']}")
    print(f"  Time        : {t_hour}:{t_min:02d}:{t_sec:02d}")
    print(f"  Saved       : {output_json}")

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
    print("  YOLO mAP@0.5 Evaluator (3-class) - Test Set")
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
        discovered_models = [(n, p) for n, p in discovered_models if n == args.model]
    elif TARGET_MODELS:
        discovered_models = [(n, p) for n, p in discovered_models if n in TARGET_MODELS]

    if not discovered_models:
        raise RuntimeError(
            f"No matching checkpoints found in {RUNS_DIR}. "
            "Ensure runs/detect/<run_name>/weights/best.pt exists."
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

    # Save CSV with European decimal format (. -> ,)
    import csv
    csv_path = OUTPUT_DIR / "best_yolo_map50_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Model", "mAP@0.5", "mAP@0.5-0.95"])
        for model_name in sorted(all_results.keys()):
            res = all_results[model_name]
            map50_str = f"{res['map_50']:.6f}".replace(".", ",")
            map50_95_str = f"{res['map_50_95']:.6f}".replace(".", ",")
            writer.writerow([model_name, map50_str, map50_95_str])
    print(f"CSV results saved: {csv_path}")

    ranking = sorted(all_results.items(), key=lambda kv: kv[1]["map_50"], reverse=True)

    print(f"\n{'=' * 70}")
    print("  mAP Ranking")
    print(f"{'=' * 70}")
    print(f"  {'Model':<15}  {'mAP@0.5':>10}  {'mAP@0.5-0.95':>14}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*14}")
    for rank, (model_name, res) in enumerate(ranking, 1):
        print(
            f"  {rank}. {model_name:<13}  {res['map_50']:>10.4f}  {res['map_50_95']:>14.4f}"
        )
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
