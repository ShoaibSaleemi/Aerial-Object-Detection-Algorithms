import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CLASS_NAMES = ["bird", "drone", "unknown"]

# Edit evaluation parameters here.
MODEL_PATH = "runs/detect/train9/weights/best.pt"  # Path to model weights (.pt) used for evaluation.
IMAGES_DIR = "dataset/validation/images"  # Directory containing validation images.
LABELS_DIR = "dataset/validation/labels"  # Directory containing YOLO-format validation label files.
IOU_THRESH = 0.5  # IoU threshold to match prediction with GT in confusion matrix counting.
CONF_THRESH = 0.70  # Minimum class confidence before applying custom WC-NMS.
NMS_THRESH = 0.50  # EIoU threshold inside Weighted-Cluster NMS (lower = more suppression).
IMGSZ = 640  # Inference image size used by letterbox preprocessing.
MAX_DET = 300  # Max raw detections retained before WC-NMS.
SAVE_PLOT_PATH = "runs/detect/train9/confusion_matrix_eval.png"  # Output path for confusion matrix image.
SAVE_METRICS_PLOT_PATH = "runs/detect/train9/metrics_table_eval.png"  # Output path for metrics table image.
SAVE_PLOTS = True  # Save confusion-matrix and metrics-table figures when True.
DEVICE = ""  # Device string: "cpu", "0", "0,1"; empty string uses default device.
VERBOSE = False  # Print per-image matching stats when True.


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


def plot_confusion(matrix, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Predicted")
    ax.set_title("Open-Set Confusion Matrix")

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

        # Thesis-style false alarm / false positive ratio from positive decisions
        pfa = safe_div(fp, tp + fp)

        # Detection probability = 1 - miss probability = TP / (TP + FN)
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
    print("\nPer-class thesis metrics:")
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
            f"{m['Precision']:>10.4f}"
            f"{m['Recall']:>10.4f}"
            f"{m['F1-score']:>10.4f}"
            f"{m['False Positive Rate']:>10.4f}"
            f"{m['Detection Probability']:>14.4f}"
        )

    print("\nMacro-average thesis metrics:")
    print(f"Precision:             {macro_metrics['Precision']:.4f}")
    print(f"Recall:                {macro_metrics['Recall']:.4f}")
    print(f"F1-score:              {macro_metrics['F1-score']:.4f}")
    print(f"False Positive Rate:   {macro_metrics['False Positive Rate']:.4f}")
    print(f"Detection Probability: {macro_metrics['Detection Probability']:.4f}")

    print("\nOpen-set summary metrics:")
    print(f"Known objects: {summary_metrics['Known objects']}")
    print(f"Unknown objects: {summary_metrics['Unknown objects']}")
    print(f"Known miss rate: {summary_metrics['Known miss rate']:.4f}")
    print(f"Unknown false alarm rate: {summary_metrics['Unknown false alarm rate']:.4f}")
    print(f"Unknown correct rejections: {summary_metrics['Unknown correct rejections']}")


def plot_metrics_table(per_class_metrics, macro_metrics, summary_metrics, save_path):
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
            f"{m['Precision']:.4f}",
            f"{m['Recall']:.4f}",
            f"{m['F1-score']:.4f}",
            f"{m['False Positive Rate']:.4f}",
            f"{m['Detection Probability']:.4f}",
        ])

    rows.append([
        "macro-avg",
        "-",
        "-",
        "-",
        "-",
        f"{macro_metrics['Precision']:.4f}",
        f"{macro_metrics['Recall']:.4f}",
        f"{macro_metrics['F1-score']:.4f}",
        f"{macro_metrics['False Positive Rate']:.4f}",
        f"{macro_metrics['Detection Probability']:.4f}",
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
        f"{summary_metrics['Unknown false alarm rate']:.4f}",
        f"{1.0 - summary_metrics['Known miss rate']:.4f}" if not np.isnan(summary_metrics["Known miss rate"]) else "nan",
    ])

    fig_h = 2.6 + 0.5 * len(rows)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    ax.set_title("Evaluation Metrics Table", fontsize=14, pad=12)

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
        f"Known miss rate: {summary_metrics['Known miss rate']:.4f}    "
        f"Unknown false alarm rate: {summary_metrics['Unknown false alarm rate']:.4f}    "
        f"Unknown correct rejections: {summary_metrics['Unknown correct rejections']}"
    )
    fig.text(0.5, 0.03, footer_text, ha="center", fontsize=10)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    label_dir = Path(LABELS_DIR)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = sorted(label_dir.glob("*.txt"))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

    model = YOLO(MODEL_PATH)
    model.model.eval()

    if DEVICE:
        model.to(DEVICE)
        device = next(model.model.parameters()).device
    else:
        device = next(model.model.parameters()).device

    valid_label_paths = []
    image_paths = []
    for label_path in label_paths:
        image_path = resolve_image_path(Path(IMAGES_DIR), label_path.stem)
        if image_path is not None:
            valid_label_paths.append(label_path)
            image_paths.append(image_path)

    if len(image_paths) == 0:
        raise ValueError(f"No validation images found in {IMAGES_DIR}")

    print(f"Running raw inference + EIoU Weighted-Cluster NMS on {len(image_paths)} validation images...")

    total_files = len(image_paths)
    processed = 0
    start_time = time.time()
    all_predictions = []

    for img_path in image_paths:
        pred_boxes, pred_labels, pred_scores = run_custom_inference(
            model=model,
            image_path=img_path,
            device=device,
            imgsz=IMGSZ,
            conf_thresh=CONF_THRESH,
            max_det=MAX_DET,
            nms_thresh=NMS_THRESH,
        )
        all_predictions.append((pred_boxes, pred_labels, pred_scores))

        processed += 1
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Progress: {processed}/{total_files} ({processed / total_files * 100:.2f}%) "
            f"Elapsed: {minutes}:{seconds:02d}",
            end="\r",
        )
    print()

    matrix, total_known, total_unknown = build_confusion_matrix(
        all_predictions,
        valid_label_paths,
        IMAGES_DIR,
        IOU_THRESH,
        VERBOSE,
    )

    print("\nConfusion matrix (rows: predicted, cols: GT):")
    print("\t" + "\t".join(CLASS_NAMES))
    for i, row in enumerate(matrix):
        print(f"{CLASS_NAMES[i]}\t" + "\t".join(str(x) for x in row))

    known_misses = matrix[2, 0] + matrix[2, 1]
    unknown_false_alarms = matrix[0, 2] + matrix[1, 2]
    unknown_correct_rejections = matrix[2, 2]

    miss_rate = known_misses / total_known if total_known > 0 else float("nan")
    false_alarm_rate = unknown_false_alarms / total_unknown if total_unknown > 0 else float("nan")

    print(f"\nKnown objects: {total_known}")
    print(f"Unknown objects: {total_unknown}")
    print(f"Known miss rate: {miss_rate:.4f} ({known_misses}/{total_known})")
    print(f"Unknown false alarm rate: {false_alarm_rate:.4f} ({unknown_false_alarms}/{total_unknown})")
    print(f"Unknown correct rejections: {unknown_correct_rejections}")

    per_class_metrics, macro_metrics, summary_metrics = compute_metrics_from_confusion(matrix)
    print_metrics_table(per_class_metrics, macro_metrics, summary_metrics)

    if SAVE_PLOTS:
        plot_confusion(matrix, SAVE_PLOT_PATH)
        print(f"Saved confusion matrix plot to {SAVE_PLOT_PATH}")

        plot_metrics_table(
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            summary_metrics=summary_metrics,
            save_path=SAVE_METRICS_PLOT_PATH,
        )
        print(f"Saved metrics table plot to {SAVE_METRICS_PLOT_PATH}")


if __name__ == "__main__":
    main()