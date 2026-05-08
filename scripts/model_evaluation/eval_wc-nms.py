import csv
import json
import random
import sys
import time
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

# Resolve repository root from scripts/algorithms/<file>.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CLASS_NAMES = ["bird", "drone", "unknown"]

# Toggle between validation and test dataset
USE_TEST_DATASET = True  # Set to True to evaluate on test dataset, False for validation

# Edit evaluation parameters here.
DATASET_SPLIT = "test" if USE_TEST_DATASET else "validation"
IMAGES_DIR = PROJECT_ROOT / "dataset" / DATASET_SPLIT / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / DATASET_SPLIT / "labels"
IOU_THRESH = 0.5
# MODEL_CONF_THRESH holds per-model thresholds tuned for standard post-NMS YOLO inference.
# They are NOT used in the WC-NMS inference path (raw scores are lower); kept for reference.
CONF_THRESH = 0.70  # Fallback post-NMS threshold (reference only).
MODEL_CONF_THRESH = {
    "yolo8n":  0.6863484706628682,
    "yolo8m":  0.7133918823950539,
    "yolo9t":  0.6724046133517759,
    "yolo10n": 0.5910035105688879,
    "yolo11n": 0.712997868833143,
    "yolo12n": 0.6838702654977842,
    "yolo26n": 0.6052508580184951,
}
# Pre-WC-NMS confidence threshold applied to raw class scores before clustering.
# Raw scores are lower than post-NMS YOLO scores, so this must be well below MODEL_CONF_THRESH.
WCNMS_CONF_THRESH = 0.30

# --- Model selection ---------------------------------------------------------
# Set a model to True to include it in evaluation, False to skip it.
# Set RUN_ALL_MODELS = True to override and run every model regardless.
RUN_ALL_MODELS = True
ENABLED_MODELS = {
    "yolo8n":  True,
    "yolo8m":  True,
    "yolo9t":  True,
    "yolo10n": True,
    "yolo11n": True,
    "yolo12n": True,
    "yolo26n": True,
}
# -----------------------------------------------------------------------------

NMS_THRESH = 0.50   
IMGSZ = 640
MAX_DET = 300

TICK_LABEL_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
CELL_VALUE_FONTSIZE = 33
PREDICTED_LABEL_PAD = -14

SAVE_PLOT = True
VERBOSE = False
DEVICE = ""

# Folder where evaluation outputs are saved.
EVAL_OUTPUT_DIR = PROJECT_ROOT / "runs" / "eval_wcnms"


def build_cache_file_path(
    detect_run_dir: Path,
    conf_thresh: float,
    iou_thresh: float,
    imgsz: int,
    nms_thresh: float,
    split: str,
) -> Path:
    cache_dir = detect_run_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"{split}_wcnms_metrics_conf_{conf_thresh:.6f}_iou_{iou_thresh:.2f}_"
        f"imgsz_{imgsz}_nms_{nms_thresh:.2f}.json"
    ).replace(".", "p")
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
    nms_thresh: float,
    matrix,
    per_class_metrics,
    macro_metrics,
    summary_metrics,
    total_known: int,
    total_unknown: int,
):
    payload = {
        "run_name": run_name,
        "method": "WC-NMS",
        "conf_thresh": float(conf_thresh),
        "iou_thresh": float(iou_thresh),
        "imgsz": int(imgsz),
        "nms_thresh": float(nms_thresh),
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
    lower_name = run_name.lower()
    if lower_name.startswith("yolo") and len(run_name) > 4:
        suffix = run_name[4:]
        if suffix and suffix[0].isdigit():
            return f"YOLOv{suffix}"
    return run_name


def resolve_image_path(images_dir: Path, stem: str):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


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

            # Same mapping logic as validation.py
            if cls == 0:
                category = 0  # bird
            elif cls == 1:
                category = 1  # drone
            else:
                category = 2  # unknown

            categories.append(category)
            boxes.append((x_center, y_center, w, h))

    return boxes, categories


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


def plot_confusion(matrix, save_path, title_prefix):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_yticklabels(CLASS_NAMES, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel("Ground Truth", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted", fontsize=AXIS_LABEL_FONTSIZE, labelpad=PREDICTED_LABEL_PAD)

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

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def safe_div(a, b):
    return a / b if b != 0 else float("nan")


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


def fmt_pct(x):
    return f"{x * 100:.2f}%" if not np.isnan(x) else "nan"


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


def preprocess_image_for_yolo(image_path: Path, imgsz: int, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    orig_shape = img_np.shape[:2]

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=True, stride=32)
    img_lb = letterbox(image=img_np)
    img_lb = img_lb.transpose((2, 0, 1))  # RGB HWC -> RGB CHW
    img_lb = np.ascontiguousarray(img_lb)

    im = torch.from_numpy(img_lb).to(device)
    im = im.float() / 255.0
    im = im.unsqueeze(0)
    return im, orig_shape


def unwrap_raw_predictions(raw_output):
    if isinstance(raw_output, torch.Tensor):
        pred = raw_output
    elif isinstance(raw_output, (list, tuple)):
        pred = None
        for item in raw_output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                pred = item
                break
            if isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, torch.Tensor) and sub.ndim == 3:
                        pred = sub
                        break
                if pred is not None:
                    break
        if pred is None:
            raise RuntimeError(f"Could not extract raw prediction tensor from output type {type(raw_output)}")
    else:
        raise RuntimeError(f"Unsupported raw model output type: {type(raw_output)}")

    return pred


def decode_raw_yolov8_predictions(raw_pred, num_classes):
    if raw_pred.ndim != 3 or raw_pred.shape[0] != 1:
        raise RuntimeError(f"Unexpected raw prediction shape: {tuple(raw_pred.shape)}")

    if raw_pred.shape[1] == 4 + num_classes:
        pred = raw_pred[0].transpose(0, 1)
    elif raw_pred.shape[2] == 4 + num_classes:
        pred = raw_pred[0]
    else:
        raise RuntimeError(
            f"Could not interpret raw prediction shape {tuple(raw_pred.shape)} for nc={num_classes}"
        )

    return pred


def box_iou_matrix_xyxy(boxes):
    x1 = torch.max(boxes[:, None, 0], boxes[None, :, 0])
    y1 = torch.max(boxes[:, None, 1], boxes[None, :, 1])
    x2 = torch.min(boxes[:, None, 2], boxes[None, :, 2])
    y2 = torch.min(boxes[:, None, 3], boxes[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
            (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp(min=1e-9)


def eiou_matrix_xyxy(boxes):
    iou = box_iou_matrix_xyxy(boxes)

    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-9)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-9)
    ctr_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    ctr_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

    dx = ctr_x[:, None] - ctr_x[None, :]
    dy = ctr_y[:, None] - ctr_y[None, :]
    d_centers = dx.pow(2) + dy.pow(2)

    dw = (widths[:, None] - widths[None, :]).pow(2)
    dh = (heights[:, None] - heights[None, :]).pow(2)

    enc_x1 = torch.min(boxes[:, None, 0], boxes[None, :, 0])
    enc_y1 = torch.min(boxes[:, None, 1], boxes[None, :, 1])
    enc_x2 = torch.max(boxes[:, None, 2], boxes[None, :, 2])
    enc_y2 = torch.max(boxes[:, None, 3], boxes[None, :, 3])

    wc = (enc_x2 - enc_x1).clamp(min=1e-9)
    hc = (enc_y2 - enc_y1).clamp(min=1e-9)

    r_eiou = d_centers / (wc.pow(2) + hc.pow(2)).clamp(min=1e-9)
    r_eiou = r_eiou + dw / wc.pow(2).clamp(min=1e-9)
    r_eiou = r_eiou + dh / hc.pow(2).clamp(min=1e-9)

    x = iou - r_eiou
    x.fill_diagonal_(0.0)
    return x


def weighted_cluster_nms_eiou_single_class(boxes, scores, thresh):
    if boxes.numel() == 0:
        return boxes, scores, torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    scores = scores[order]

    x = eiou_matrix_xyxy(boxes)
    x = torch.triu(x, diagonal=1)

    n = boxes.shape[0]
    b_prev = torch.ones(n, device=boxes.device, dtype=torch.float32)
    c_final = None
    b_final = b_prev.clone()

    for _ in range(n):
        a_t = torch.diag(b_prev)
        c_t = a_t @ x
        g = c_t.max(dim=0).values
        b_t = (g < thresh).float()

        c_final = c_t
        b_final = b_t

        if torch.equal(b_t, b_prev):
            break
        b_prev = b_t

    keep_mask = b_final.bool()

    if keep_mask.sum() == 0:
        return (
            torch.empty((0, 4), device=boxes.device),
            torch.empty((0,), device=boxes.device),
            torch.empty((0,), dtype=torch.long, device=boxes.device),
        )

    c_prime = c_final + torch.eye(n, device=boxes.device, dtype=c_final.dtype)
    c_prime = c_prime * scores.unsqueeze(1)

    weights = c_prime[:, keep_mask].transpose(0, 1)
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    merged_boxes = weights @ boxes / denom

    kept_scores = scores[keep_mask]
    kept_indices = order[keep_mask]

    return merged_boxes, kept_scores, kept_indices


def run_custom_inference(model, image_path: Path, device, imgsz, conf_thresh, max_det, nms_thresh):
    im, orig_shape = preprocess_image_for_yolo(image_path, imgsz, device)

    with torch.no_grad():
        raw_output = model.model(im)

    raw_pred = unwrap_raw_predictions(raw_output)
    pred = decode_raw_yolov8_predictions(raw_pred, num_classes=len(model.names))

    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    if cls_scores.shape[1] < 2:
        raise RuntimeError(
            f"Model appears to have fewer than 2 classes in output: {cls_scores.shape[1]}"
        )

    confs, clses = cls_scores.max(dim=1)

    valid_mask = (clses < 2) & (confs >= conf_thresh)
    box_xywh = box_xywh[valid_mask]
    confs = confs[valid_mask]
    clses = clses[valid_mask]

    if box_xywh.numel() == 0:
        return [], [], []

    box_xyxy = ops.xywh2xyxy(box_xywh)

    if box_xyxy.shape[0] > max_det:
        topk = torch.argsort(confs, descending=True)[:max_det]
        box_xyxy = box_xyxy[topk]
        confs = confs[topk]
        clses = clses[topk]

    final_boxes = []
    final_scores = []
    final_labels = []

    for class_id in [0, 1]:
        mask = clses == class_id
        if mask.sum() == 0:
            continue

        cls_boxes = box_xyxy[mask]
        cls_scores = confs[mask]

        kept_boxes, kept_scores, _ = weighted_cluster_nms_eiou_single_class(
            cls_boxes, cls_scores, nms_thresh
        )

        if kept_boxes.numel() == 0:
            continue

        kept_boxes = ops.scale_boxes(
            img1_shape=im.shape[2:],
            boxes=kept_boxes.clone(),
            img0_shape=orig_shape,
        )

        for b, s in zip(kept_boxes.cpu(), kept_scores.cpu()):
            final_boxes.append([float(v) for v in b.tolist()])
            final_scores.append(float(s.item()))
            final_labels.append(int(class_id))

    if len(final_scores) > 0:
        order = np.argsort(-np.array(final_scores))
        final_boxes = [final_boxes[i] for i in order]
        final_scores = [final_scores[i] for i in order]
        final_labels = [final_labels[i] for i in order]

    return final_boxes, final_labels, final_scores


def run_yolo_inference_before_wc_nms(model, image_path: Path, device, imgsz, conf_thresh, max_det):
    """Run the same decoded YOLO predictions before applying custom WC-NMS."""
    im, orig_shape = preprocess_image_for_yolo(image_path, imgsz, device)

    with torch.no_grad():
        raw_output = model.model(im)

    raw_pred = unwrap_raw_predictions(raw_output)
    pred = decode_raw_yolov8_predictions(raw_pred, num_classes=len(model.names))

    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    if cls_scores.shape[1] < 2:
        raise RuntimeError(
            f"Model appears to have fewer than 2 classes in output: {cls_scores.shape[1]}"
        )

    confs, clses = cls_scores.max(dim=1)

    valid_mask = (clses < 2) & (confs >= conf_thresh)
    box_xywh = box_xywh[valid_mask]
    confs = confs[valid_mask]
    clses = clses[valid_mask]

    if box_xywh.numel() == 0:
        return [], [], []

    box_xyxy = ops.xywh2xyxy(box_xywh)

    if box_xyxy.shape[0] > max_det:
        topk = torch.argsort(confs, descending=True)[:max_det]
        box_xyxy = box_xyxy[topk]
        confs = confs[topk]
        clses = clses[topk]

    box_xyxy = ops.scale_boxes(
        img1_shape=im.shape[2:],
        boxes=box_xyxy.clone(),
        img0_shape=orig_shape,
    )

    final_boxes = [[float(v) for v in b.tolist()] for b in box_xyxy.cpu()]
    final_labels = [int(c.item()) for c in clses.cpu()]
    final_scores = [float(s.item()) for s in confs.cpu()]

    return final_boxes, final_labels, final_scores


def build_confusion_matrix(all_predictions, label_paths, images_dir, iou_thresh, verbose):
    matrix = np.zeros((3, 3), dtype=int)
    total_known = 0
    total_unknown = 0

    for image_idx, label_path in enumerate(sorted(label_paths)):
        image_name = label_path.stem
        image_path = resolve_image_path(Path(images_dir), image_name)
        if image_path is None:
            continue

        gt_boxes_xywh, gt_labels = load_label_file(label_path)

        with Image.open(image_path) as img:
            width, height = img.size

        gt_boxes = [xywhn_to_xyxy(box, width, height) for box in gt_boxes_xywh]

        pred_boxes, pred_labels, _ = all_predictions[image_idx]
        assignments, _ = match_predictions(
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
            print(f"{image_name}: GT {len(gt_labels)}, pred {len(pred_boxes)}, matched {len(assignments)}")

    return matrix, total_known, total_unknown


def main():
    detect_root_dir = PROJECT_ROOT / "runs" / "detect"
    available_runs = sorted(
        [
            path.name
            for path in detect_root_dir.iterdir()
            if path.is_dir() and path.name.lower().startswith("yolo")
        ]
    )

    if len(available_runs) == 0:
        raise ValueError(f"No yolo* folders found in {detect_root_dir}")

    if len(sys.argv) > 1:
        run_names = [sys.argv[1]]
        if run_names[0] not in available_runs:
            available_text = ", ".join(available_runs)
            raise ValueError(
                f"Unknown folder '{run_names[0]}'. Choose one from runs/detect: {available_text}"
            )
    elif RUN_ALL_MODELS:
        run_names = available_runs
    else:
        run_names = [name for name in available_runs if ENABLED_MODELS.get(name, False)]
        if not run_names:
            raise ValueError("No models enabled. Set RUN_ALL_MODELS=True or enable at least one in ENABLED_MODELS.")

    label_dir = LABELS_DIR
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = list(label_dir.glob("*.txt"))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

    image_paths = []
    valid_label_paths = []
    for label_path in sorted(label_paths):
        image_path = resolve_image_path(IMAGES_DIR, label_path.stem)
        if image_path is not None:
            image_paths.append(image_path)
            valid_label_paths.append(label_path)

    if len(image_paths) == 0:
        raise ValueError(f"No {DATASET_SPLIT} images found in {IMAGES_DIR}")

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_model_rows = []  # Accumulates rows for the combined summary CSV

    for run_name in run_names:
        detect_run_dir = detect_root_dir / run_name
        model_pt = detect_run_dir / "weights" / "best.pt"
        if not model_pt.exists():
            print(f"[SKIP] {run_name}: no weights/best.pt found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Model: {run_name}")
        print(f"{'=' * 60}")

        run_display_name = format_run_display_name(run_name)
        conf_thresh = WCNMS_CONF_THRESH
        cache_path = build_cache_file_path(detect_run_dir, conf_thresh, IOU_THRESH, IMGSZ, NMS_THRESH, DATASET_SPLIT)
        save_plot_path = EVAL_OUTPUT_DIR / f"{run_name}_WC-NMS.png"

        cached = load_eval_cache(cache_path)
        if cached is not None:
            print(f"Using cached WC-NMS evaluation for {run_name} (pre-nms-conf={conf_thresh:.2f})")
            matrix = cached["matrix"]
            total_known = cached["total_known"]
            total_unknown = cached["total_unknown"]
            per_class_metrics = cached["per_class_metrics"]
            macro_metrics = cached["macro_metrics"]
            summary_metrics = cached["summary_metrics"]
        else:
            print(f"Running WC-NMS inference on {len(image_paths)} {DATASET_SPLIT} images (pre-nms-conf={conf_thresh:.2f})...")
            model = YOLO(str(model_pt))
            model.model.eval()

            if DEVICE:
                model.to(DEVICE)
                device = next(model.model.parameters()).device
            else:
                device = next(model.model.parameters()).device

            total_files = len(image_paths)
            processed = 0
            start_time = time.time()
            all_predictions_wcnms = []

            for img_path in image_paths:
                pred_boxes, pred_labels, pred_scores = run_custom_inference(
                    model=model,
                    image_path=img_path,
                    device=device,
                    imgsz=IMGSZ,
                    conf_thresh=conf_thresh,
                    max_det=MAX_DET,
                    nms_thresh=NMS_THRESH,
                )
                all_predictions_wcnms.append((pred_boxes, pred_labels, pred_scores))

                processed += 1
                elapsed = time.time() - start_time
                minutes, seconds = divmod(int(elapsed), 60)
                print(
                    f"Progress: {processed}/{total_files} ({processed / total_files * 100:.2f}%) Elapsed: {minutes}:{seconds:02d}",
                    end="\r",
                )
            print()

            matrix, total_known, total_unknown = build_confusion_matrix(
                all_predictions_wcnms,
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
                nms_thresh=NMS_THRESH,
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

        # Accumulate rows for combined summary CSV
        for m in per_class_metrics:
            all_model_rows.append([
                run_display_name + " WC-NMS",
                m["class"],
                f"{m['Precision'] * 100:.2f}".replace('.', ',') if not np.isnan(m["Precision"]) else "nan",
                f"{m['Recall'] * 100:.2f}".replace('.', ',') if not np.isnan(m["Recall"]) else "nan",
                f"{m['F1-score'] * 100:.2f}".replace('.', ',') if not np.isnan(m["F1-score"]) else "nan",
            ])
        all_model_rows.append([
            run_display_name + " WC-NMS",
            "average",
            f"{macro_metrics['Precision'] * 100:.2f}".replace('.', ',') if not np.isnan(macro_metrics["Precision"]) else "nan",
            f"{macro_metrics['Recall'] * 100:.2f}".replace('.', ',') if not np.isnan(macro_metrics["Recall"]) else "nan",
            f"{macro_metrics['F1-score'] * 100:.2f}".replace('.', ',') if not np.isnan(macro_metrics["F1-score"]) else "nan",
        ])

        if SAVE_PLOT:
            plot_confusion(matrix, save_plot_path, run_display_name)
            print(f"Saved confusion matrix plot to {save_plot_path}")

    # Save combined summary CSV for all models
    if all_model_rows:
        combined_csv_path = EVAL_OUTPUT_DIR / "all_models_metrics.csv"
        wcnms_model_names = {row[0] for row in all_model_rows}
        preserved_rows = []
        if combined_csv_path.exists():
            with combined_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file)
                next(reader, None)  # skip header
                for row in reader:
                    if row and row[0] not in wcnms_model_names:
                        preserved_rows.append(row)
        with combined_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Class", "Precision", "Recall", "F1"])
            writer.writerows(all_model_rows)
            writer.writerows(preserved_rows)
        print(f"\nSaved combined metrics CSV to {combined_csv_path}")


if __name__ == "__main__":
    main()