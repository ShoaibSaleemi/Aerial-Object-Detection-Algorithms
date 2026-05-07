from itertools import zip_longest
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import random
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# This script is intended for open-set evaluation of a Faster R-CNN checkpoint
# trained on known classes. Validation labels may include `unknown`, and any
# prediction outside the known label set is folded into `unknown` for metrics.
CLASS_NAMES = ["bird", "drone", "unknown"]

# Edit these parameters directly before running this script.
CONFIG = {
    "model": str(PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "fasterrcnn_epoch_50.pt"),
    "images": str(PROJECT_ROOT / "dataset" / "validation" / "images"),
    "labels": str(PROJECT_ROOT / "dataset" / "validation" / "labels"),
    "iou_thresh": 0.5,
    "conf_thresh": 0.996286,
    "save_plot": str(PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "confusion_matrix_val.png"),
    "save_metrics_plot": str(PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "metrics_table_val.png"),
    "save_plot_enabled": True,
    "verbose": False,
    "device": "auto",  # "cpu", "cuda", or "auto"
}

TICK_LABEL_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
CELL_VALUE_FONTSIZE = 33


def build_cache_file_path(run_dir: Path, conf_thresh: float, iou_thresh: float) -> Path:
    cache_dir = run_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"metrics_conf_{conf_thresh:.6f}_iou_{iou_thresh:.2f}.json"
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
    }


def save_eval_cache(
    cache_path: Path,
    checkpoint_path: Path,
    conf_thresh: float,
    iou_thresh: float,
    matrix,
    per_class_metrics,
    macro_metrics,
    summary_metrics,
):
    payload = {
        "checkpoint": str(checkpoint_path),
        "conf_thresh": float(conf_thresh),
        "iou_thresh": float(iou_thresh),
        "labels_dir": str(CONFIG["labels"]),
        "images_dir": str(CONFIG["images"]),
        "matrix": matrix.tolist(),
        "per_class_metrics": per_class_metrics,
        "macro_metrics": macro_metrics,
        "summary_metrics": summary_metrics,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_model_display_name(model_path: Path) -> str:
    """Return a clean model family name for figure titles."""
    stem = model_path.stem.lower()
    if stem.startswith("fasterrcnn"):
        return "Faster R-CNN"
    return model_path.stem.replace("_", " ")


def load_label_file(label_path: Path):
    """Load YOLO format labels and convert to pixel coordinates."""
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


def xywhn_to_xyxy(box, img_width, img_height):
    """Convert normalized YOLO xywh to pixel xyxy."""
    x_center, y_center, w, h = box
    x1 = (x_center - w / 2.0) * img_width
    y1 = (y_center - h / 2.0) * img_height
    x2 = (x_center + w / 2.0) * img_width
    y2 = (y_center + h / 2.0) * img_height
    return [x1, y1, x2, y2]


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes in xyxy format."""
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
    """Match ground truth and predictions using greedy IoU matching."""
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


def build_model(checkpoint_path: Path, device="cpu"):
    """Load Faster R-CNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer num_classes from checkpoint
    cls_score_weight = checkpoint["model_state_dict"]["roi_heads.box_predictor.cls_score.weight"]
    num_classes = cls_score_weight.shape[0]
    
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_inference(model, image_path: Path, conf_thresh: float, device="cpu"):
    """Run Faster R-CNN inference on a single image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = (
        torch.from_numpy(np.array(image, dtype="uint8"))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .to(device)
    )

    with torch.no_grad():
        predictions = model([image_tensor])

    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    filtered_boxes = []
    filtered_labels = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= conf_thresh:
            filtered_boxes.append(box)
            # Shift from torchvision (0=background) back to YOLO format (0=bird, 1=drone)
            shifted_label = label - 1
            if shifted_label in (0, 1):
                filtered_labels.append(shifted_label)
            else:
                filtered_labels.append(2)

    return filtered_boxes, filtered_labels


def build_confusion_matrix(
    model, image_paths, label_paths, conf_thresh, iou_thresh, verbose, device
):
    """Build confusion matrix from inference results."""
    matrix = np.zeros((3, 3), dtype=int)
    total_known = 0
    total_unknown = 0
    total_files = len(image_paths)
    start_time = time.time()

    for image_idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        image_name = label_path.stem

        gt_boxes_xywh, gt_labels = load_label_file(label_path)
        if len(gt_labels) == 0 and verbose:
            print(f"Skipping {image_name} because it has no labels.")
        
        # Print progress
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Progress: {image_idx + 1}/{total_files} ({(image_idx + 1) / total_files * 100:.2f}%) "
            f"Elapsed: {minutes}:{seconds:02d}",
            end="\r",
        )

        with Image.open(image_path) as img:
            width, height = img.size
        gt_boxes = [xywhn_to_xyxy(box, width, height) for box in gt_boxes_xywh]

        pred_boxes, pred_labels = run_inference(model, image_path, conf_thresh, device)

        assignments, used_pred = match_predictions(
            gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh
        )

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

    print()  # Newline after progress bar
    return matrix, total_known, total_unknown


def plot_confusion(matrix, save_path, title_prefix):
    """Plot and save confusion matrix."""
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
            text_color = "white" if i == j else "black"
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color=text_color,
                fontsize=CELL_VALUE_FONTSIZE,
            )

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

        pfa = safe_div(fp, tp + fp)
        p_success = recall

        per_class_metrics.append({
            "class": CLASS_NAMES[c],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "False Positive Rate": pfa,
            "Detection Probability": p_success,
        })

    macro_metrics = {
        "Precision": np.nanmean([m["Precision"] for m in per_class_metrics]),
        "Recall": np.nanmean([m["Recall"] for m in per_class_metrics]),
        "F1-score": np.nanmean([m["F1-score"] for m in per_class_metrics]),
        "False Positive Rate": np.nanmean([m["False Positive Rate"] for m in per_class_metrics]),
        "Detection Probability": np.nanmean([m["Detection Probability"] for m in per_class_metrics]),
    }

    total_known = int(matrix[:, 0].sum() + matrix[:, 1].sum())
    total_unknown = int(matrix[:, 2].sum())

    known_misses = int(matrix[2, 0] + matrix[2, 1])
    unknown_false_alarms = int(matrix[0, 2] + matrix[1, 2])
    unknown_correct_rejections = int(matrix[2, 2])

    summary_metrics = {
        "Known objects": total_known,
        "Unknown objects": total_unknown,
        "Known miss rate": safe_div(known_misses, total_known),
        "Unknown false alarm rate": safe_div(unknown_false_alarms, total_unknown),
        "Unknown correct rejections": unknown_correct_rejections,
    }

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


def plot_metrics_table(per_class_metrics, macro_metrics, summary_metrics, save_path, title_prefix):
    rows = []
    columns = ["Class", "Precision", "Recall", "F1-score"]

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

    fig_h = 2.6 + 0.5 * len(rows)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.axis("off")
    ax.set_title(f"{title_prefix} Evaluation Metrics Table", fontsize=14, pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    save_path = Path(save_path)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    model_path = CONFIG["model"]
    images_dir = CONFIG["images"]
    labels_dir = CONFIG["labels"]
    iou_thresh = CONFIG["iou_thresh"]
    conf_thresh = CONFIG["conf_thresh"]
    save_plot = Path(CONFIG["save_plot"])
    save_metrics_plot = Path(CONFIG["save_metrics_plot"])
    save_plot_enabled = CONFIG["save_plot_enabled"]
    verbose = CONFIG["verbose"]
    device = CONFIG["device"]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model_display_name = format_model_display_name(checkpoint_path)
    run_dir = checkpoint_path.parent
    cache_path = build_cache_file_path(run_dir, conf_thresh, iou_thresh)
    save_plot_path = save_plot
    save_metrics_plot_path = save_metrics_plot

    label_dir = Path(labels_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = list(label_dir.glob("*.txt"))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

    image_paths = []
    valid_label_paths = []
    for label_path in sorted(label_paths):
        image_name = label_path.stem
        image_path = Path(images_dir) / f"{image_name}.jpg"
        if not image_path.exists():
            for ext in [".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = Path(images_dir) / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path.exists():
            image_paths.append(image_path)
            valid_label_paths.append(label_path)

    if len(image_paths) == 0:
        raise ValueError(f"No validation images found in {images_dir}")

    cached = load_eval_cache(cache_path)
    if cached is not None:
        print(f"Using cached evaluation (conf={conf_thresh:.4f}, iou={iou_thresh:.2f})")
        matrix = cached["matrix"]
        per_class_metrics = cached["per_class_metrics"]
        macro_metrics = cached["macro_metrics"]
        summary_metrics = cached["summary_metrics"]
    else:
        print(f"Loading model from {checkpoint_path}...")
        model = build_model(checkpoint_path, device=device)

        print(f"Running inference on {len(image_paths)} validation images (conf={conf_thresh:.4f})...")

        matrix, _, _ = build_confusion_matrix(
            model,
            image_paths,
            valid_label_paths,
            conf_thresh,
            iou_thresh,
            verbose,
            device,
        )

        per_class_metrics, macro_metrics, summary_metrics = compute_metrics_from_confusion(matrix)
        save_eval_cache(
            cache_path=cache_path,
            checkpoint_path=checkpoint_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            matrix=matrix,
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            summary_metrics=summary_metrics,
        )
        print(f"Saved evaluation cache: {cache_path}")

    print_confusion_and_metrics_side_by_side(matrix, per_class_metrics, macro_metrics)
    print()

    if save_plot_enabled:
        plot_confusion(matrix, save_plot_path, model_display_name)
        print(f"Saved confusion matrix plot to {save_plot_path}")

        plot_metrics_table(
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            summary_metrics=summary_metrics,
            save_path=save_metrics_plot_path,
            title_prefix=model_display_name,
        )
        print(f"Saved metrics table plot to {save_metrics_plot_path}")


if __name__ == "__main__":
    main()
