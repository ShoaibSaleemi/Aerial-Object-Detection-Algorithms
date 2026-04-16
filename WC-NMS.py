import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 with class-aware EIoU Weighted-Cluster NMS on validation data."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train9/weights/best.pt",
        help="Path to the trained YOLO model weights.",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/validation/images",
        help="Directory containing validation images.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/validation/labels",
        help="Directory containing validation labels.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for GT/pred matching in confusion matrix.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.70,
        help="Confidence threshold for candidate detections before custom NMS.",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.50,
        help="EIoU threshold used inside Weighted-Cluster NMS.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum raw detections kept before custom NMS.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="runs/detect/train9/confusion_matrix_eval.png",
        help="Path to save confusion matrix visualization.",
    )
    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Do not save the confusion matrix figure.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help='Device to run on, e.g. "cpu", "0", "0,1". Empty = auto.',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image statistics during evaluation.",
    )
    return parser.parse_args()


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

            # Same mapping logic as your current validation.py
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


def preprocess_image_for_yolo(image_path: Path, imgsz: int, device: torch.device):
    """
    Returns:
        im_tensor: (1, 3, H, W) float tensor in [0,1]
        orig_bgr: original image as numpy array in BGR-like shape is not required
        orig_shape: (h, w)
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # RGB, HWC
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
    """
    Tries to extract the main detection tensor from Ultralytics forward output.
    Expected common shape after extraction:
        (bs, no, num_preds) or (bs, num_preds, no)
    """
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
    """
    Converts raw prediction tensor to shape (num_preds, 4 + num_classes)
    with boxes in xywh format.

    For YOLOv8 detect models, a common inference shape is (1, 4+nc, N).
    """
    if raw_pred.ndim != 3 or raw_pred.shape[0] != 1:
        raise RuntimeError(f"Unexpected raw prediction shape: {tuple(raw_pred.shape)}")

    # Try to standardize to (N, 4 + nc)
    if raw_pred.shape[1] == 4 + num_classes:
        pred = raw_pred[0].transpose(0, 1)  # (no, N) -> (N, no)
    elif raw_pred.shape[2] == 4 + num_classes:
        pred = raw_pred[0]  # (N, no)
    else:
        raise RuntimeError(
            f"Could not interpret raw prediction shape {tuple(raw_pred.shape)} for nc={num_classes}"
        )

    return pred


def box_iou_matrix_xyxy(boxes):
    """
    boxes: (N, 4) xyxy
    returns IoU matrix (N, N)
    """
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
    """
    EIoU matrix based on the uploaded paper:
        X = MIoU - REIoU
    where REIoU uses center distance, width difference, height difference,
    normalized by enclosing box dimensions. Values can be < 0.
    """
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
    """
    Class-aware EIoU Weighted-Cluster NMS for one class.

    boxes: (N, 4) xyxy
    scores: (N,)
    returns:
        kept_boxes, kept_scores, kept_indices_in_sorted_order
    """
    if boxes.numel() == 0:
        return boxes, scores, torch.empty((0,), dtype=torch.long, device=boxes.device)

    """ Sort by scores descending """
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    scores = scores[order]

    """ Build the upper-triangular EIoU matrix """
    x = eiou_matrix_xyxy(boxes)
    x = torch.triu(x, diagonal=1)  # upper triangular only

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

    # Weighted coordinates:
    # The paper's formula uses C' * B / Repmat4(sum_i C'(i,:)).
    # To make weighted merging usable for kept leaders too, we include self-links with identity.
    c_prime = c_final + torch.eye(n, device=boxes.device, dtype=c_final.dtype)
    c_prime = c_prime * scores.unsqueeze(1)  # multiply each row i by score_i

    weights = c_prime[:, keep_mask].transpose(0, 1)  # (#kept, N)
    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    merged_boxes = weights @ boxes / denom

    kept_scores = scores[keep_mask]
    kept_indices = order[keep_mask]

    return merged_boxes, kept_scores, kept_indices


def run_custom_inference(model, image_path: Path, args, device):
    """
    Runs raw forward pass, decodes predictions, applies class-aware EIoU Weighted-Cluster NMS.
    Returns:
        pred_boxes: list of [x1,y1,x2,y2] in original image coordinates
        pred_labels: list of int labels in {0,1}
        pred_scores: list of float scores
    """
    im, orig_shape = preprocess_image_for_yolo(image_path, args.imgsz, device)

    with torch.no_grad():
        raw_output = model.model(im)

    raw_pred = unwrap_raw_predictions(raw_output)
    pred = decode_raw_yolov8_predictions(raw_pred, num_classes=len(model.names))

    # YOLOv8 detect head commonly outputs [x, y, w, h, class_scores...]
    # Unlike older YOLO variants, there is typically no separate objectness channel here.
    box_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    if cls_scores.shape[1] < 2:
        raise RuntimeError(
            f"Model appears to have fewer than 2 classes in output: {cls_scores.shape[1]}"
        )

    confs, clses = cls_scores.max(dim=1)

    # Keep only bird/drone predictions; unknown is GT-only, same as your validation.py logic.
    valid_mask = (clses < 2) & (confs >= args.conf_thresh)
    box_xywh = box_xywh[valid_mask]
    confs = confs[valid_mask]
    clses = clses[valid_mask]

    if box_xywh.numel() == 0:
        return [], [], []

    # Convert xywh -> xyxy in resized/letterboxed image coordinates
    box_xyxy = ops.xywh2xyxy(box_xywh)

    # Limit raw candidates before custom NMS
    if box_xyxy.shape[0] > args.max_det:
        topk = torch.argsort(confs, descending=True)[: args.max_det]
        box_xyxy = box_xyxy[topk]
        confs = confs[topk]
        clses = clses[topk]

    final_boxes = []
    final_scores = []
    final_labels = []

    # Class-aware custom NMS: process bird and drone separately
    for class_id in [0, 1]:
        mask = clses == class_id
        if mask.sum() == 0:
            continue

        cls_boxes = box_xyxy[mask]
        cls_scores = confs[mask]

        kept_boxes, kept_scores, _ = weighted_cluster_nms_eiou_single_class(
            cls_boxes, cls_scores, args.nms_thresh
        )

        if kept_boxes.numel() == 0:
            continue

        # Scale boxes back to original image size
        kept_boxes = ops.scale_boxes(
            img1_shape=im.shape[2:],
            boxes=kept_boxes.clone(),
            img0_shape=orig_shape,
        )

        for b, s in zip(kept_boxes.cpu(), kept_scores.cpu()):
            final_boxes.append([float(v) for v in b.tolist()])
            final_scores.append(float(s.item()))
            final_labels.append(int(class_id))

    # Sort final outputs by score descending
    if len(final_scores) > 0:
        order = np.argsort(-np.array(final_scores))
        final_boxes = [final_boxes[i] for i in order]
        final_scores = [final_scores[i] for i in order]
        final_labels = [final_labels[i] for i in order]

    return final_boxes, final_labels, final_scores


def build_confusion_matrix(all_predictions, label_paths, args):
    matrix = np.zeros((3, 3), dtype=int)
    total_known = 0
    total_unknown = 0

    for image_idx, label_path in enumerate(sorted(label_paths)):
        image_name = label_path.stem
        image_path = resolve_image_path(Path(args.images), image_name)
        if image_path is None:
            continue

        gt_boxes_xywh, gt_labels = load_label_file(label_path)

        with Image.open(image_path) as img:
            width, height = img.size

        gt_boxes = [xywhn_to_xyxy(box, width, height) for box in gt_boxes_xywh]

        pred_boxes, pred_labels, _ = all_predictions[image_idx]
        assignments, used_pred = match_predictions(
            gt_boxes, gt_labels, pred_boxes, pred_labels, args.iou_thresh
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

        if args.verbose and len(gt_labels) > 0:
            print(
                f"{image_name}: GT {len(gt_labels)}, pred {len(pred_boxes)}, matched {len(assignments)}"
            )

    return matrix, total_known, total_unknown


def main():
    args = parse_args()

    label_dir = Path(args.labels)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = sorted(label_dir.glob("*.txt"))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

    model = YOLO(args.model)
    model.model.eval()

    # Device selection
    if args.device:
        model.to(args.device)
        device = next(model.model.parameters()).device
    else:
        device = next(model.model.parameters()).device

    valid_label_paths = []
    image_paths = []
    for label_path in label_paths:
        image_path = resolve_image_path(Path(args.images), label_path.stem)
        if image_path is not None:
            valid_label_paths.append(label_path)
            image_paths.append(image_path)

    if len(image_paths) == 0:
        raise ValueError(f"No validation images found in {args.images}")

    print(f"Running raw inference + EIoU Weighted-Cluster NMS on {len(image_paths)} validation images...")

    total_files = len(image_paths)
    processed = 0
    start_time = time.time()
    all_predictions = []

    for img_path in image_paths:
        pred_boxes, pred_labels, pred_scores = run_custom_inference(
            model=model,
            image_path=img_path,
            args=args,
            device=device,
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

    matrix, total_known, total_unknown = build_confusion_matrix(all_predictions, valid_label_paths, args)

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

    if not args.no_save_plot:
        plot_confusion(matrix, args.save_plot)
        print(f"Saved confusion matrix plot to {args.save_plot}")


if __name__ == "__main__":
    main()