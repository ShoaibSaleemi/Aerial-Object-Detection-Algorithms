"""Compute mAP@0.5 for Faster R-CNN checkpoints on test dataset.

For each discovered checkpoint in runs/fasterrcnn/train, this script runs
inference at conf=0.001 on the test images, then computes per-class
Average Precision at IoU 0.5 using 101-point COCO-style interpolation.

  mAP@0.5 = mean of per-class APs (classes: bird, drone, unknown)

Class mapping convention in this project:
- Ground truth labels: 0=bird, 1=drone, anything else -> unknown(2)
- Faster R-CNN outputs: 0=background, 1=bird, 2=drone, ...
  so predictions are shifted by -1, then remapped to {0,1,2}.

Usage:
    python "tools/eval_fasterrcnn_map50.py"
    python "tools/eval_fasterrcnn_map50.py" --model best
    python "tools/eval_fasterrcnn_map50.py" --model fasterrcnn_epoch_50
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
IMAGES_DIR = PROJECT_ROOT / "dataset" / "test" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "test" / "labels"
CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "fasterrcnn" / "train"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "fasterrcnn" / "map50"
AGGREGATE_JSON = OUTPUT_DIR / "best_fasterrcnn_map50_test.json"

CLASS_NAMES = ["bird", "drone", "unknown"]
N_CLASSES = len(CLASS_NAMES)

# Very low confidence to collect full PR curves.
CONF_INFER = 0.001
IOU_THRESH = 0.5

IOU_THRESHOLDS_95 = np.linspace(0.5, 0.95, 10)  # [0.50, 0.55, ..., 0.95]

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


def discover_checkpoints(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    checkpoints = []
    for ckpt in sorted(checkpoint_dir.glob("*.pt")):
        checkpoints.append((ckpt.stem, ckpt))
    return checkpoints


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


def image_to_tensor(image_path: Path, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return (
        torch.from_numpy(np.array(image, dtype="uint8"))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .to(device)
    )


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
    device: str,
    start_time: float,
) -> tuple[float, float, list[dict]]:
    total_files = len(image_paths)

    per_class_preds_all_ious: list[list[list[tuple[float, int]]]] = [
        [[] for _ in range(len(IOU_THRESHOLDS_95))] for _ in range(N_CLASSES)
    ]
    per_class_n_gt = [0] * N_CLASSES

    for idx, (image_path, label_path) in enumerate(zip(image_paths, valid_label_paths), 1):
        gt_boxes_xywh, gt_labels = load_label_file(label_path)
        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes_xyxy = [xywhn_to_xyxy(b, width, height) for b in gt_boxes_xywh]
        for lbl in gt_labels:
            per_class_n_gt[lbl] += 1

        image_tensor = image_to_tensor(image_path, device)
        with torch.no_grad():
            pred = model([image_tensor])[0]

        pred_boxes_xyxy: list[list[float]] = []
        pred_confs: list[float] = []
        pred_classes: list[int] = []

        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if float(score) < CONF_INFER:
                continue
            shifted = int(label) - 1
            cls_id = shifted if shifted in (0, 1) else 2
            pred_classes.append(cls_id)
            pred_confs.append(float(score))
            pred_boxes_xyxy.append(list(box))

        for c in range(N_CLASSES):
            c_gt_boxes = [gt_boxes_xyxy[i] for i, lbl in enumerate(gt_labels) if lbl == c]

            c_pred_items = sorted(
                [(pred_confs[j], pred_boxes_xyxy[j]) for j in range(len(pred_classes)) if pred_classes[j] == c],
                key=lambda x: x[0],
                reverse=True,
            )

            # For each IoU threshold, compute TP/FP
            for iou_idx, iou_t in enumerate(IOU_THRESHOLDS_95):
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

                    if best_iou >= iou_t and best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                        per_class_preds_all_ious[c][iou_idx].append((conf_val, 1))
                    else:
                        per_class_preds_all_ious[c][iou_idx].append((conf_val, 0))

        pct = idx / total_files * 100.0
        elapsed = time.time() - start_time
        t_hour, t_rem = divmod(int(elapsed), 3600)
        t_min, t_sec = divmod(t_rem, 60)
        _print_inline_status(
            f"  {idx}/{total_files} ({pct:.1f}%) | Elapsed: {t_hour}:{t_min:02d}:{t_sec:02d}"
        )

    _finish_inline_status_line()

    per_class_results = []
    aps50 = []
    aps50_95 = []

    for c in range(N_CLASSES):
        n_gt = per_class_n_gt[c]

        if n_gt == 0:
            per_class_results.append({
                "class": CLASS_NAMES[c],
                "n_gt": 0,
                "ap50": None,
                "ap50_95": None,
            })
            continue

        # Compute AP@0.5 (first IoU threshold)
        preds_50 = per_class_preds_all_ious[c][0]
        if not preds_50:
            ap50 = 0.0
            aps50.append(0.0)
        else:
            preds_50.sort(key=lambda x: x[0], reverse=True)
            tp_arr = np.array([p[1] for p in preds_50], dtype=np.float64)
            fp_arr = 1.0 - tp_arr
            cum_tp = np.cumsum(tp_arr)
            cum_fp = np.cumsum(fp_arr)
            recalls = cum_tp / n_gt
            precisions = cum_tp / (cum_tp + cum_fp)
            ap50 = compute_ap_101(recalls, precisions)
            aps50.append(ap50)

        # Compute AP at each IoU threshold for mAP@0.5-0.95
        aps_all = []
        for iou_idx in range(len(IOU_THRESHOLDS_95)):
            preds_at_iou = per_class_preds_all_ious[c][iou_idx]
            if not preds_at_iou:
                aps_all.append(0.0)
            else:
                preds_at_iou.sort(key=lambda x: x[0], reverse=True)
                tp_arr = np.array([p[1] for p in preds_at_iou], dtype=np.float64)
                fp_arr = 1.0 - tp_arr
                cum_tp = np.cumsum(tp_arr)
                cum_fp = np.cumsum(fp_arr)
                recalls = cum_tp / n_gt
                precisions = cum_tp / (cum_tp + cum_fp)
                ap = compute_ap_101(recalls, precisions)
                aps_all.append(ap)

        ap50_95 = float(np.mean(aps_all)) if aps_all else 0.0
        aps50_95.append(ap50_95)

        per_class_results.append({
            "class": CLASS_NAMES[c],
            "n_gt": n_gt,
            "ap50": round(float(ap50), 6) if ap50 is not None else None,
            "ap50_95": round(float(ap50_95), 6),
        })

    map50 = float(np.mean(aps50)) if aps50 else 0.0
    map50_95 = float(np.mean(aps50_95)) if aps50_95 else 0.0
    return map50, map50_95, per_class_results


def evaluate_single_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    device: str,
) -> dict | None:
    if not checkpoint_path.exists():
        print(f"  [SKIP] {model_name}: checkpoint not found at {checkpoint_path}")
        return None

    print(f"\n{'=' * 70}")
    print(f"  Evaluating checkpoint : {model_name}")
    print(f"  Path                  : {checkpoint_path}")
    print(f"{'=' * 70}")

    model = build_model(checkpoint_path, device)
    start_time = time.time()

    map50, map50_95, per_class_results = compute_map50(
        model=model,
        image_paths=image_paths,
        valid_label_paths=valid_label_paths,
        device=device,
        start_time=start_time,
    )

    elapsed = time.time() - start_time
    t_hour, t_rem = divmod(int(elapsed), 3600)
    t_min, t_sec = divmod(t_rem, 60)

    result = {
        "checkpoint": model_name,
        "checkpoint_path": str(checkpoint_path),
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
    print(f"  Saved   : {output_json}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mAP@0.5 for Faster R-CNN checkpoints (3-class: bird, drone, unknown)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best",
        help="Evaluate a single checkpoint stem (example: best or fasterrcnn_epoch_50). Omit to evaluate all *.pt files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: auto, cpu, cuda",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")
    if not LABELS_DIR.is_dir():
        raise FileNotFoundError(f"Labels dir not found: {LABELS_DIR}")
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {CHECKPOINT_DIR}")

    print("=" * 70)
    print("  Faster R-CNN mAP@0.5 Evaluator (3-class) - Test Set")
    print("=" * 70)
    print(f"  Images dir     : {IMAGES_DIR}")
    print(f"  Labels dir     : {LABELS_DIR}")
    print(f"  Checkpoint dir : {CHECKPOINT_DIR}")
    print(f"  Inference conf : {CONF_INFER}  (low to collect full PR curve)")
    print(f"  IoU threshold  : {IOU_THRESH}")
    print(f"  Device         : {device}")

    print("\nLoading test images and labels...")
    image_paths, valid_label_paths = find_validation_pairs(IMAGES_DIR, LABELS_DIR)
    print(f"  {len(image_paths)} image-label pairs loaded")

    print("\nDiscovering checkpoints...")
    discovered = discover_checkpoints(CHECKPOINT_DIR)
    if not discovered:
        raise RuntimeError(f"No .pt checkpoints found in {CHECKPOINT_DIR}")

    if args.model is not None:
        selected = [(n, p) for n, p in discovered if n == args.model]
        if not selected:
            available = ", ".join(n for n, _ in discovered)
            raise ValueError(f"Unknown model '{args.model}'. Available: {available}")
        checkpoints = selected
    else:
        checkpoints = discovered

    print(f"  {len(checkpoints)} checkpoint(s) to evaluate:")
    for ckpt_name, ckpt_path in checkpoints:
        print(f"    - {ckpt_name}: {ckpt_path}")

    all_results = {}
    for ckpt_name, ckpt_path in checkpoints:
        try:
            result = evaluate_single_checkpoint(
                model_name=ckpt_name,
                checkpoint_path=ckpt_path,
                image_paths=image_paths,
                valid_label_paths=valid_label_paths,
                device=device,
            )
            if result is not None:
                all_results[ckpt_name] = result
        except Exception as exc:
            _finish_inline_status_line()
            print(f"\n  [WARN] Failed evaluating {ckpt_name}: {exc}")
            continue

    if not all_results:
        raise RuntimeError("No checkpoint produced an evaluation result.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "generated_at_epoch": int(time.time()),
        "settings": {
            "iou_thresh": float(IOU_THRESH),
            "conf_infer": float(CONF_INFER),
            "images_dir": str(IMAGES_DIR),
            "labels_dir": str(LABELS_DIR),
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "device": device,
        },
        "checkpoints": all_results,
    }

    with AGGREGATE_JSON.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nAggregate results saved: {AGGREGATE_JSON}")

    # Save CSV with European decimal format (. -> ,)
    import csv
    csv_path = OUTPUT_DIR / "best_fasterrcnn_map50_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Checkpoint", "mAP@0.5", "mAP@0.5-0.95"])
        for ckpt_name in sorted(all_results.keys()):
            res = all_results[ckpt_name]
            map50_str = f"{res['map_50']:.6f}".replace(".", ",")
            map50_95_str = f"{res['map_50_95']:.6f}".replace(".", ",")
            writer.writerow([ckpt_name, map50_str, map50_95_str])
    print(f"CSV results saved: {csv_path}")

    ranking = sorted(all_results.items(), key=lambda kv: kv[1]["map_50"], reverse=True)

    print(f"\n{'=' * 70}")
    print("  mAP@0.5 Ranking")
    print(f"{'=' * 70}")
    print(f"  {'Checkpoint':<20}  {'mAP@0.5':>10}  {'mAP@0.5-0.95':>14}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*14}")
    for rank, (name, res) in enumerate(ranking, 1):
        print(f"  {rank}. {name:<18}  {res['map_50']:>10.4f}  {res['map_50_95']:>14.4f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
