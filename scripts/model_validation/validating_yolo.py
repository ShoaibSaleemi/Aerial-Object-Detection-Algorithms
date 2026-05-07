import csv
import json
from pathlib import Path
import sys
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import questionary
import time
from PIL import Image
from ultralytics import YOLO
import random
import torch    

PROJECT_ROOT = Path(__file__).resolve().parents[2]

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CLASS_NAMES = ["bird", "drone", "unknown"]

# Edit evaluation parameters here.
IMAGES_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "validation" / "labels"
IOU_THRESH = 0.5
CONF_THRESH = 0.70
MODEL_CONF_THRESH = {
    "yolo8n": 0.6863484706628682,
    "yolo8m": 0.7133918823950539,
    "yolo9t": 0.6724046133517759,
    "yolo10n": 0.5910035105688879,
    "yolo11n": 0.712997868833143,
    "yolo12n": 0.6838702654977842,
    "yolo26n": 0.6052508580184951,
}
IMGSZ = 640

TICK_LABEL_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 16
CELL_VALUE_FONTSIZE = 20

SAVE_PLOT = True
VERBOSE = False  # Print per-image matching/debug details during evaluation when True.


def build_cache_file_path(detect_run_dir: Path, conf_thresh: float, iou_thresh: float, imgsz: int) -> Path:
    cache_dir = detect_run_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"metrics_conf_{conf_thresh:.6f}_iou_{iou_thresh:.2f}_imgsz_{imgsz}.json"
        .replace(".", "p")
    )
    return cache_dir / cache_name


def load_eval_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    matrix_data = payload.get("matrix")
    if not isinstance(matrix_data, list):
        return None

    matrix = np.array(matrix_data, dtype=int)
    return {
        "matrix": matrix,
        "per_class_metrics": payload.get("per_class_metrics", []),
        "macro_metrics": payload.get("macro_metrics", {}),
        "summary_metrics": payload.get("summary_metrics", {}),
        "total_known": int(payload.get("total_known", 0)),
        "total_unknown": int(payload.get("total_unknown", 0)),
    }


def save_eval_cache(
    cache_path: Path,
    run_name: str,
    conf_thresh: float,
    iou_thresh: float,
    imgsz: int,
    matrix,
    per_class_metrics,
    macro_metrics,
    summary_metrics,
    total_known: int,
    total_unknown: int,
):
    payload = {
        "run_name": run_name,
        "conf_thresh": float(conf_thresh),
        "iou_thresh": float(iou_thresh),
        "imgsz": int(imgsz),
        "labels_dir": str(LABELS_DIR),
        "images_dir": str(IMAGES_DIR),
        "total_known": int(total_known),
        "total_unknown": int(total_unknown),
        "matrix": matrix.tolist(),
        "per_class_metrics": per_class_metrics,
        "macro_metrics": macro_metrics,
        "summary_metrics": summary_metrics,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_run_display_name(run_name: str) -> str:
    """Convert run folder names like yolo8n to display names like YOLOv8n."""
    lower_name = run_name.lower()
    if lower_name.startswith("yolo") and len(run_name) > 4:
        suffix = run_name[4:]
        if suffix and suffix[0].isdigit():
            return f"YOLOv{suffix}"
    return run_name


def load_label_file(label_path: Path):
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
            if cls == 0:
                category = 0
            elif cls == 1:
                category = 1
            else:
                category = 2
            categories.append(category)
            boxes.append((x_center, y_center, w, h))

    return boxes, categories


def yolo_xywh_to_xyxy(box):
    x_center, y_center, w, h = box
    x1 = x_center - w / 2.0
    y1 = y_center - h / 2.0
    x2 = x_center + w / 2.0
    y2 = y_center + h / 2.0
    return [x1, y1, x2, y2]


def xywhn_to_xyxy(box, img_width, img_height):
    x_center, y_center, w, h = box
    x1 = (x_center - w / 2.0) * img_width
    y1 = (y_center - h / 2.0) * img_height
    x2 = (x_center + w / 2.0) * img_width
    y2 = (y_center + h / 2.0) * img_height
    return [x1, y1, x2, y2]


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh):
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_gt == 0:
        return {}, set()

    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(n_pred):
            ious[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])

    used_gt = set()
    used_pred = set()
    assignments = {}

    while True:
        if ious.size == 0:
            break
        max_idx = np.unravel_index(np.argmax(ious), ious.shape)
        max_iou = ious[max_idx]
        if max_iou < iou_thresh:
            break
        gt_idx, pred_idx = max_idx
        assignments[gt_idx] = pred_idx
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        ious[gt_idx, :] = -1.0
        ious[:, pred_idx] = -1.0

    return assignments, used_pred


def build_confusion_matrix(results, label_paths, images_dir, iou_thresh, verbose):
    matrix = np.zeros((3, 3), dtype=int)
    total_known = 0
    total_unknown = 0
    total_images = len(label_paths)

    for image_idx, label_path in enumerate(sorted(label_paths)):
        image_name = label_path.stem
        image_path = (Path(images_dir) / f"{image_name}.jpg")
        if not image_path.exists():
            image_path = None
            for ext in [".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = Path(images_dir) / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is None:
            continue

        gt_boxes_xywh, gt_labels = load_label_file(label_path)
        if len(gt_labels) == 0 and verbose:
            print(f"Skipping {image_name} because it has no labels.")

        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes = [xywhn_to_xyxy(box, width, height) for box in gt_boxes_xywh]

        result = results[image_idx]
        pred_boxes = []
        pred_labels = []
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                pred_labels.append(int(cls) if int(cls) in (0, 1) else 2)
                pred_boxes.append(list(box))

        assignments, used_pred = match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh)

        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_label in (0, 1):
                total_known += 1
            else:
                total_unknown += 1

            if gt_idx in assignments:
                pred_idx = assignments[gt_idx]
                pred_label = pred_labels[pred_idx]
                matrix[pred_label, gt_label] += 1
            else:
                matrix[2, gt_label] += 1

        if verbose and len(gt_labels) > 0:
            print(
                f"{image_name}: GT {len(gt_labels)}, pred {len(pred_boxes)}, matched {len(assignments)}"
            )

        print(
            f"Matrix progress: {image_idx + 1}/{total_images} ({(image_idx + 1) / total_images * 100:.2f}%)",
            end="\r",
        )

    print()
    return matrix, total_known, total_unknown


def plot_confusion(matrix, save_path, title_prefix):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_yticklabels(CLASS_NAMES, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel("Ground Truth", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted", fontsize=AXIS_LABEL_FONTSIZE)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_color = "white" if i == 2 and j == 2 else "black"
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color=text_color,
                fontsize=CELL_VALUE_FONTSIZE,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([
        cbar_pos.x0,
        cbar_pos.y0 - 0.2,
        cbar_pos.width,
        cbar_pos.height,
    ])
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def safe_div(a, b):
    return a / b if b != 0 else float("nan")


def fmt_pct(x):
    return f"{x * 100:.2f}%" if not np.isnan(x) else "nan"


def compute_metrics_from_confusion(matrix):
    """
    rows = predicted, cols = ground truth
    """
    n_classes = matrix.shape[0]
    total = int(matrix.sum())

    per_class_metrics = []

    for c in range(n_classes):
        tp = int(matrix[c, c])
        fp = int(matrix[c, :].sum() - tp)
        fn = int(matrix[:, c].sum() - tp)
        tn = int(total - tp - fp - fn)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if not (
            np.isnan(precision) or np.isnan(recall)
        ) else float("nan")

        per_class_metrics.append({
            "class": CLASS_NAMES[c],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        })

    macro_metrics = {
        "Precision": np.nanmean([m["Precision"] for m in per_class_metrics]),
        "Recall": np.nanmean([m["Recall"] for m in per_class_metrics]),
        "F1-score": np.nanmean([m["F1-score"] for m in per_class_metrics]),
    }

    summary_metrics = {}

    return per_class_metrics, macro_metrics, summary_metrics


def build_confusion_table_lines(matrix):
    lines = ["Confusion matrix (rows: predicted, cols: GT):"]
    row_label_w = 10
    col_w = 8
    for i, row in enumerate(matrix):
        lines.append(
            f"{CLASS_NAMES[i]:<{row_label_w}}"
            + "".join(f"{int(x):>{col_w}}" for x in row)
        )
    lines.append(" " * row_label_w + "".join(f"{name:>{col_w}}" for name in CLASS_NAMES))
    return lines


def build_metrics_table_lines(per_class_metrics, macro_metrics):
    lines = ["Per-class metrics:"]
    header = (
        f"{'Class':<10}"
        f"{'Prec':>10}{'Recall':>10}{'F1':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for m in per_class_metrics:
        lines.append(
            f"{m['class']:<10}"
            f"{fmt_pct(m['Precision']):>10}"
            f"{fmt_pct(m['Recall']):>10}"
            f"{fmt_pct(m['F1-score']):>10}"
        )

    lines.append("-" * len(header))
    red = "\033[38;2;255;42;0m"
    reset = "\033[0m"
    lines.append(
        f"{'macro-avg':<10}"
        + red
        + f"{fmt_pct(macro_metrics['Precision']):>10}"
        + f"{fmt_pct(macro_metrics['Recall']):>10}"
        + f"{fmt_pct(macro_metrics['F1-score']):>10}"
        + reset
    )
    return lines


def print_confusion_and_metrics_side_by_side(matrix, per_class_metrics, macro_metrics):
    left_lines = build_confusion_table_lines(matrix)
    right_lines = build_metrics_table_lines(per_class_metrics, macro_metrics)

    left_width = max(len(line) for line in left_lines)
    gap = 4
    for left, right in zip_longest(left_lines, right_lines, fillvalue=""):
        print(f"{left:<{left_width}}{' ' * gap}{right}")


def save_metrics_table_csv(per_class_metrics, macro_metrics, save_path):
    rows = []
    for m in per_class_metrics:
        rows.append([
            m["class"],
            fmt_pct(m["Precision"]),
            fmt_pct(m["Recall"]),
            fmt_pct(m["F1-score"]),
        ])

    rows.append([
        "macro-avg",
        fmt_pct(macro_metrics["Precision"]),
        fmt_pct(macro_metrics["Recall"]),
        fmt_pct(macro_metrics["F1-score"]),
    ])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Class", "Precision", "Recall", "F1-score"])
        writer.writerows(rows)


def main():
    detect_root_dir = PROJECT_ROOT / "runs" / "detect"
    available_runs = sorted([path.name for path in detect_root_dir.iterdir() if path.is_dir()])

    if len(available_runs) == 0:
        raise ValueError(f"No folders found in {detect_root_dir}")

    if len(sys.argv) > 1:
        run_name = sys.argv[1]
        if run_name not in available_runs:
            available_text = ", ".join(available_runs)
            raise ValueError(
                f"Unknown folder '{run_name}'. Choose one from runs/detect: {available_text}"
            )
    else:
        run_name = questionary.select(
            "Choose a folder from runs/detect:",
            choices=available_runs,
        ).ask()
        if not run_name:
            raise ValueError("No folder selected from runs/detect")

    detect_run_dir = detect_root_dir / run_name
    run_display_name = format_run_display_name(run_name)
    conf_thresh = MODEL_CONF_THRESH.get(run_name, CONF_THRESH)
    cache_path = build_cache_file_path(detect_run_dir, conf_thresh, IOU_THRESH, IMGSZ)
    model_path = str(detect_run_dir / "weights" / "best.pt")
    save_plot_path = str(detect_run_dir / "confusion_matrix_eval.png")
    save_metrics_csv_path = str(detect_run_dir / "metrics_table_eval.csv")

    label_dir = LABELS_DIR
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = list(label_dir.glob("*.txt"))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

    model = YOLO(model_path)
    image_paths = []
    valid_label_paths = []
    for label_path in sorted(label_paths):
        image_name = label_path.stem
        image_path = IMAGES_DIR / f"{image_name}.jpg"
        if not image_path.exists():
            for ext in [".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = IMAGES_DIR / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path.exists():
            image_paths.append(str(image_path))
            valid_label_paths.append(label_path)

    if len(image_paths) == 0:
        raise ValueError(f"No validation images found in {IMAGES_DIR}")

    cached = load_eval_cache(cache_path)
    if cached is not None:
        print(
            f"Using cached evaluation for {run_name} "
            f"(conf={conf_thresh:.4f}, iou={IOU_THRESH:.2f}, imgsz={IMGSZ})"
        )
        matrix = cached["matrix"]
        total_known = cached["total_known"]
        total_unknown = cached["total_unknown"]
        per_class_metrics = cached["per_class_metrics"]
        macro_metrics = cached["macro_metrics"]
        summary_metrics = cached["summary_metrics"]
    else:
        print(f"Running inference on {len(image_paths)} validation images (conf={conf_thresh:.4f})...")
        # Inference with progress bar
        total_files = len(image_paths)
        processed = 0
        start_time = time.time()
        results = []
        for img_path in image_paths:
            result = model.predict(
                source=img_path,
                conf=conf_thresh,
                imgsz=IMGSZ,
                verbose=False,
            )
            results.append(result[0] if isinstance(result, list) else result)
            processed += 1
            elapsed = time.time() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            print(f"Progress: {processed}/{total_files} ({processed / total_files * 100:.2f}%) Elapsed: {minutes}:{seconds:02d}", end='\r')
        print()  # Newline after progress bar

        matrix, total_known, total_unknown = build_confusion_matrix(
            results,
            valid_label_paths,
            IMAGES_DIR,
            IOU_THRESH,
            VERBOSE,
        )
        per_class_metrics, macro_metrics, summary_metrics = compute_metrics_from_confusion(matrix)

        save_eval_cache(
            cache_path=cache_path,
            run_name=run_name,
            conf_thresh=conf_thresh,
            iou_thresh=IOU_THRESH,
            imgsz=IMGSZ,
            matrix=matrix,
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            summary_metrics=summary_metrics,
            total_known=total_known,
            total_unknown=total_unknown,
        )
        print(f"Saved evaluation cache: {cache_path}")

    print_confusion_and_metrics_side_by_side(matrix, per_class_metrics, macro_metrics)
    print()

    if SAVE_PLOT:
        plot_confusion(matrix, save_plot_path, run_display_name)
        print(f"Saved confusion matrix plot to {save_plot_path}")

        save_metrics_table_csv(
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            save_path=save_metrics_csv_path,
        )
        print(f"Saved metrics table CSV to {save_metrics_csv_path}")


if __name__ == "__main__":
    main()

