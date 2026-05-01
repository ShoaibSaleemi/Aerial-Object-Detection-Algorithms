from pathlib import Path

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
    "conf_thresh": 0.5,
    "save_plot": str(PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "confusion_matrix_val.png"),
    "save_metrics_plot": str(PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "metrics_table_val.png"),
    "save_plot_enabled": True,
    "verbose": False,
    "device": "auto",  # "cpu", "cuda", or "auto"
}


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
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title_prefix} Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
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


def print_metrics_table(per_class_metrics, macro_metrics, summary_metrics):
    print("\nPer-class metrics:")
    header = (
        f"{'Class':<10}"
        f"{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>8}"
        f"{'Prec':>10}{'Recall':>10}{'F1':>10}{'Pfa':>10}{'P(success)':>14}"
    )
    print(header)
    print("-" * len(header))

    for m in per_class_metrics:
        print(
            f"{m['class']:<10}"
            f"{m['TP']:>8}{m['FP']:>8}{m['FN']:>8}{m['TN']:>8}"
            f"{fmt_pct(m['Precision']):>10}"
            f"{fmt_pct(m['Recall']):>10}"
            f"{fmt_pct(m['F1-score']):>10}"
            f"{fmt_pct(m['False Positive Rate']):>10}"
            f"{fmt_pct(m['Detection Probability']):>14}"
        )

    print("\nMacro-average metrics:")
    print(f"Precision:             {fmt_pct(macro_metrics['Precision'])}")
    print(f"Recall:                {fmt_pct(macro_metrics['Recall'])}")
    print(f"F1-score:              {fmt_pct(macro_metrics['F1-score'])}")
    print(f"False Positive Rate:   {fmt_pct(macro_metrics['False Positive Rate'])}")
    print(f"Detection Probability: {fmt_pct(macro_metrics['Detection Probability'])}")

    print("\nOpen-set summary metrics:")
    print(f"Known objects: {summary_metrics['Known objects']}")
    print(f"Unknown objects: {summary_metrics['Unknown objects']}")
    print(f"Known miss rate: {fmt_pct(summary_metrics['Known miss rate'])}")
    print(f"Unknown false alarm rate: {fmt_pct(summary_metrics['Unknown false alarm rate'])}")
    print(f"Unknown correct rejections: {summary_metrics['Unknown correct rejections']}")


def plot_metrics_table(per_class_metrics, macro_metrics, summary_metrics, save_path, title_prefix):
    rows = []
    columns = [
        "Class", "TP", "FP", "FN", "TN",
        "Precision", "Recall", "F1-score", "Pfa", "P(success)"
    ]

    for m in per_class_metrics:
        rows.append([
            m["class"],
            m["TP"],
            m["FP"],
            m["FN"],
            m["TN"],
            fmt_pct(m["Precision"]),
            fmt_pct(m["Recall"]),
            fmt_pct(m["F1-score"]),
            fmt_pct(m["False Positive Rate"]),
            fmt_pct(m["Detection Probability"]),
        ])

    rows.append([
        "macro-avg",
        "-",
        "-",
        "-",
        "-",
        fmt_pct(macro_metrics["Precision"]),
        fmt_pct(macro_metrics["Recall"]),
        fmt_pct(macro_metrics["F1-score"]),
        fmt_pct(macro_metrics["False Positive Rate"]),
        fmt_pct(macro_metrics["Detection Probability"]),
    ])

    rows.append([
        "open-set",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        fmt_pct(summary_metrics["Unknown false alarm rate"]),
        fmt_pct(1.0 - summary_metrics["Known miss rate"]) if not np.isnan(summary_metrics["Known miss rate"]) else "nan",
    ])

    fig_h = 2.6 + 0.5 * len(rows)
    fig, ax = plt.subplots(figsize=(13, fig_h))
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

    footer_text = (
        f"Known objects: {summary_metrics['Known objects']}    "
        f"Unknown objects: {summary_metrics['Unknown objects']}    "
        f"Known miss rate: {fmt_pct(summary_metrics['Known miss rate'])}    "
        f"Unknown false alarm rate: {fmt_pct(summary_metrics['Unknown false alarm rate'])}    "
        f"Unknown correct rejections: {summary_metrics['Unknown correct rejections']}"
    )
    fig.text(0.5, 0.03, footer_text, ha="center", fontsize=10)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    model_path = CONFIG["model"]
    images_dir = CONFIG["images"]
    labels_dir = CONFIG["labels"]
    iou_thresh = CONFIG["iou_thresh"]
    conf_thresh = CONFIG["conf_thresh"]
    save_plot = CONFIG["save_plot"]
    save_metrics_plot = CONFIG["save_metrics_plot"]
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

    print(f"Loading model from {checkpoint_path}...")
    model = build_model(checkpoint_path, device=device)

    print(f"Running inference on {len(image_paths)} validation images...")

    matrix, _, _ = build_confusion_matrix(
        model,
        image_paths,
        valid_label_paths,
        conf_thresh,
        iou_thresh,
        verbose,
        device,
    )

    print("\nConfusion matrix (rows: predicted, cols: GT):")
    print("\t" + "\t".join(CLASS_NAMES))
    for i, row in enumerate(matrix):
        print(f"{CLASS_NAMES[i]}\t" + "\t".join(str(x) for x in row))

    per_class_metrics, macro_metrics, summary_metrics = compute_metrics_from_confusion(matrix)
    print_metrics_table(per_class_metrics, macro_metrics, summary_metrics)
    print()

    if save_plot_enabled:
        plot_confusion(matrix, save_plot, model_display_name)
        print(f"Saved confusion matrix plot to {save_plot}")

        plot_metrics_table(
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            summary_metrics=summary_metrics,
            save_path=save_metrics_plot,
            title_prefix=model_display_name,
        )
        print(f"Saved metrics table plot to {save_metrics_plot}")


if __name__ == "__main__":
    main()
