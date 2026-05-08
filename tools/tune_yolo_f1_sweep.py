"""
Fast YOLO confidence tuning by cache + sweep (no Optuna).

For each YOLO checkpoint, this script:
1. Runs one low-confidence inference pass (CONF_INFER) and caches predictions.
2. Sweeps confidence thresholds over a fixed range.
3. Picks the threshold with peak macro-averaged F1.

Usage:
    python "tools/tune_yolo_f1_sweep.py"
    python "tools/tune_yolo_f1_sweep.py" --model yolo8m --steps 300
    python "tools/tune_yolo_f1_sweep.py" --conf-min 0.70 --conf-max 0.95 --batch 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "validation" / "labels"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
OUTPUT_DIR = RUNS_DIR / "tune_yolo_f1_sweep"
AGGREGATE_JSON = OUTPUT_DIR / "best_yolo_f1_sweep.json"

CLASS_NAMES = ["bird", "drone", "unknown"]
N_CLASSES = len(CLASS_NAMES)
IOU_THRESH = 0.5
IMGSZ = 640

# One inference pass to cache predictions; sweep is then CPU-only.
CONF_INFER = 0.001
CONF_MIN = 0.55
CONF_MAX = 0.85
N_STEPS = 1000
BATCH_SIZE = 16

IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]

_INLINE_STATUS_LEN = 0


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


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def load_label_file(label_path: Path) -> tuple[list, list]:
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
        max_iou = ious[max_idx]
        if max_iou < iou_thresh:
            break

        gt_idx, pred_idx = max_idx
        assignments[gt_idx] = pred_idx
        ious[gt_idx, :] = -1.0
        ious[:, pred_idx] = -1.0

    return assignments


def compute_metrics_from_confusion(matrix: np.ndarray) -> dict:
    per_class = []
    for c in range(N_CLASSES):
        tp = int(matrix[c, c])
        fp = int(matrix[c, :].sum() - tp)
        fn = int(matrix[:, c].sum() - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = (
            safe_div(2 * precision * recall, precision + recall)
            if not (np.isnan(precision) or np.isnan(recall))
            else float("nan")
        )

        per_class.append(
            {
                "class": CLASS_NAMES[c],
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
            }
        )

    macro = {
        "Precision": float(np.nanmean([m["Precision"] for m in per_class])),
        "Recall": float(np.nanmean([m["Recall"] for m in per_class])),
        "F1-score": float(np.nanmean([m["F1-score"] for m in per_class])),
    }
    return macro


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
        if model_name.lower().startswith("yolo"):
            models.append((model_name, best_path))
    return models


def cache_model_predictions(
    model,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    start_time: float,
    batch_size: int,
) -> list[dict]:
    total_files = len(image_paths)
    cached = []

    for start in range(0, total_files, batch_size):
        end = min(start + batch_size, total_files)
        batch_imgs = image_paths[start:end]
        batch_labels = valid_label_paths[start:end]

        results = model.predict(
            source=[str(p) for p in batch_imgs],
            conf=CONF_INFER,
            imgsz=IMGSZ,
            verbose=False,
            batch=batch_size,
        )

        for local_idx, (result, label_path) in enumerate(zip(results, batch_labels), 1):
            gt_boxes_xywh, gt_labels = load_label_file(label_path)

            img_h, img_w = result.orig_shape
            gt_boxes_xyxy = [xywhn_to_xyxy(b, img_w, img_h) for b in gt_boxes_xywh]

            pred_entries = []
            if hasattr(result, "boxes") and len(result.boxes) > 0:
                for box, conf, cls in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                ):
                    cls_id = int(cls)
                    pred_entries.append(
                        (
                            float(conf),
                            cls_id if cls_id in (0, 1) else 2,
                            list(box),
                        )
                    )

            cached.append(
                {
                    "gt_boxes": gt_boxes_xyxy,
                    "gt_labels": gt_labels,
                    "preds": pred_entries,
                }
            )

            processed = start + local_idx
            pct = processed / total_files * 100.0
            elapsed = time.time() - start_time
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            _print_inline_status(
                f"  Caching {processed}/{total_files} ({pct:.1f}%) | Elapsed: {h}:{m:02d}:{s:02d}"
            )

    _finish_inline_status_line()
    return cached


def compute_f1_from_cache(cached_data: list[dict], conf_thresh: float) -> tuple[float, float, float]:
    matrix = np.zeros((3, 3), dtype=int)

    for entry in cached_data:
        gt_boxes = entry["gt_boxes"]
        gt_labels = entry["gt_labels"]

        filtered = [p for p in entry["preds"] if p[0] >= conf_thresh]
        pred_boxes = [p[2] for p in filtered]
        pred_labels = [p[1] for p in filtered]

        assignments = match_predictions(gt_boxes, pred_boxes, IOU_THRESH)
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx in assignments:
                pred_label = pred_labels[assignments[gt_idx]]
                matrix[pred_label, gt_label] += 1
            else:
                matrix[2, gt_label] += 1

    macro = compute_metrics_from_confusion(matrix)

    precision = float(macro["Precision"])
    recall = float(macro["Recall"])
    f1 = float(macro["F1-score"])

    if np.isnan(precision):
        precision = 0.0
    if np.isnan(recall):
        recall = 0.0
    if np.isnan(f1):
        f1 = 0.0

    return precision, recall, f1


def sweep_thresholds(
    cached_data: list[dict],
    conf_min: float,
    conf_max: float,
    n_steps: int,
    start_time: float,
) -> list[dict]:
    thresholds = np.linspace(conf_min, conf_max, n_steps)
    sweep_results = []

    for idx, conf_thresh in enumerate(thresholds, 1):
        precision, recall, f1 = compute_f1_from_cache(cached_data, float(conf_thresh))
        sweep_results.append(
            {
                "conf_thresh": float(conf_thresh),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

        pct = idx / n_steps * 100.0
        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        _print_inline_status(
            f"  Sweep {idx}/{n_steps} ({pct:.1f}%) | conf={conf_thresh:.4f} "
            f"| F1={f1:.4f} P={precision:.4f} R={recall:.4f} | Elapsed: {h}:{m:02d}:{s:02d}"
        )

    _finish_inline_status_line()
    return sweep_results


def tune_single_model(
    model_name: str,
    model_path: Path,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    conf_min: float,
    conf_max: float,
    n_steps: int,
    batch_size: int,
) -> dict | None:
    if not model_path.exists():
        print(f"  [SKIP] {model_name}: checkpoint not found at {model_path}")
        return None

    print(f"\n{'=' * 70}")
    print(f"  Sweeping model: {model_name}")
    print(f"  Checkpoint   : {model_path}")
    print(f"{'=' * 70}")

    model = YOLO(str(model_path))
    start_time = time.time()

    print("  Caching inference results (one-time pass at conf=0.001)...")
    cached_data = cache_model_predictions(
        model=model,
        image_paths=image_paths,
        valid_label_paths=valid_label_paths,
        start_time=start_time,
        batch_size=batch_size,
    )
    print(f"  Cache ready ({len(cached_data)} images). Sweeping thresholds...")

    sweep_results = sweep_thresholds(
        cached_data=cached_data,
        conf_min=conf_min,
        conf_max=conf_max,
        n_steps=n_steps,
        start_time=start_time,
    )

    best = max(sweep_results, key=lambda x: x["f1"])

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    print("  --- Best ---")
    print(f"  conf_thresh: {best['conf_thresh']:.6f}")
    print(f"  F1:         {best['f1']:.4f}")
    print(f"  Precision:  {best['precision']:.4f}")
    print(f"  Recall:     {best['recall']:.4f}")
    print(f"  Time:       {h}:{m:02d}:{s:02d}")

    output = {
        "model": model_name,
        "model_path": str(model_path),
        "iou_thresh": IOU_THRESH,
        "imgsz": IMGSZ,
        "conf_infer": CONF_INFER,
        "conf_min": conf_min,
        "conf_max": conf_max,
        "steps": int(n_steps),
        "batch_size": int(batch_size),
        "eval_time_seconds": round(elapsed, 1),
        "best": best,
        "sweep": sweep_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_json = OUTPUT_DIR / f"best_{model_name}.json"
    with model_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {model_json}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast YOLO confidence tuning by cached inference + threshold sweep"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Tune one model by run folder name (example: yolo8m). Omit for all YOLO runs.",
    )
    parser.add_argument("--steps", type=int, default=N_STEPS, help=f"Sweep steps (default: {N_STEPS})")
    parser.add_argument("--conf-min", type=float, default=CONF_MIN, help=f"Min conf (default: {CONF_MIN})")
    parser.add_argument("--conf-max", type=float, default=CONF_MAX, help=f"Max conf (default: {CONF_MAX})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help=f"Inference batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("--steps must be >= 2")
    if not (0.0 <= args.conf_min < args.conf_max <= 1.0):
        raise ValueError("Require 0 <= conf-min < conf-max <= 1")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    print("=" * 70)
    print("  YOLO F1 Threshold Sweep (cache + sweep)")
    print("=" * 70)
    print(f"  Images dir   : {IMAGES_DIR}")
    print(f"  Labels dir   : {LABELS_DIR}")
    print(f"  Runs dir     : {RUNS_DIR}")
    print(f"  Infer conf   : {CONF_INFER}")
    print(f"  Sweep range  : [{args.conf_min}, {args.conf_max}] in {args.steps} steps")
    print(f"  Batch size   : {args.batch}")

    image_paths, valid_label_paths = find_validation_pairs(IMAGES_DIR, LABELS_DIR)
    print(f"\nLoaded {len(image_paths)} validation image-label pairs")

    discovered = discover_models(RUNS_DIR)
    if not discovered:
        raise ValueError(f"No YOLO checkpoints found under {RUNS_DIR}/*/weights/best.pt")

    if args.model is not None:
        selected = [(name, path) for name, path in discovered if name == args.model]
        if not selected:
            available = ", ".join(name for name, _ in discovered)
            raise ValueError(f"Unknown model '{args.model}'. Available: {available}")
        models_to_run = selected
    else:
        models_to_run = discovered

    all_results = []
    for model_name, model_path in models_to_run:
        result = tune_single_model(
            model_name=model_name,
            model_path=model_path,
            image_paths=image_paths,
            valid_label_paths=valid_label_paths,
            conf_min=args.conf_min,
            conf_max=args.conf_max,
            n_steps=args.steps,
            batch_size=args.batch,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No models were tuned.")
        return

    aggregate_payload = {
        "settings": {
            "iou_thresh": IOU_THRESH,
            "imgsz": IMGSZ,
            "conf_infer": CONF_INFER,
            "conf_min": args.conf_min,
            "conf_max": args.conf_max,
            "steps": int(args.steps),
            "batch_size": int(args.batch),
        },
        "models": [
            {
                "model": r["model"],
                "model_path": r["model_path"],
                "best": r["best"],
                "eval_time_seconds": r["eval_time_seconds"],
            }
            for r in all_results
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with AGGREGATE_JSON.open("w", encoding="utf-8") as f:
        json.dump(aggregate_payload, f, indent=2)

    print(f"\nSaved aggregate summary: {AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
