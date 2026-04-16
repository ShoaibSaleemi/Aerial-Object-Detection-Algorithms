import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
import random
import torch    

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CLASS_NAMES = ["bird", "drone", "unknown"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 closed-set training with open-set unknown handling on validation data."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train9/weights/best.pt", # model to evulate
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
        help="IoU threshold for matching predictions to ground truth.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.70,
        help="Confidence threshold for inference.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="runs/detect/train9/confusion_matrix_eval.png", # location to save the confusion matrix plot
        help="Path to save confusion matrix visualization.",
    )
    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Do not save the confusion matrix figure.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image statistics during evaluation.",
    )
    return parser.parse_args()


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


def build_confusion_matrix(results, label_paths, args):
    matrix = np.zeros((3, 3), dtype=int)
    total_known = 0
    total_unknown = 0
    total_images = len(label_paths)

    for image_idx, label_path in enumerate(sorted(label_paths)):
        image_name = label_path.stem
        image_path = (Path(args.images) / f"{image_name}.jpg")
        if not image_path.exists():
            image_path = None
            for ext in [".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = Path(args.images) / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is None:
            continue

        gt_boxes_xywh, gt_labels = load_label_file(label_path)
        if len(gt_labels) == 0 and args.verbose:
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

        assignments, used_pred = match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, args.iou_thresh)

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


def main():
    args = parse_args()
    label_dir = Path(args.labels)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = list(label_dir.glob("*.txt"))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

    model = YOLO(args.model)
    image_paths = []
    valid_label_paths = []
    for label_path in sorted(label_paths):
        image_name = label_path.stem
        image_path = Path(args.images) / f"{image_name}.jpg"
        if not image_path.exists():
            for ext in [".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = Path(args.images) / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path.exists():
            image_paths.append(str(image_path))
            valid_label_paths.append(label_path)

    if len(image_paths) == 0:
        raise ValueError(f"No validation images found in {args.images}")

    print(f"Running inference on {len(image_paths)} validation images...")
    # Inference with progress bar
    total_files = len(image_paths)
    processed = 0
    start_time = time.time()
    results = []
    for img_path in image_paths:
        result = model.predict(
            source=img_path,
            conf=args.conf_thresh,
            imgsz=args.imgsz,
            verbose=False,
        )
        results.append(result[0] if isinstance(result, list) else result)
        processed += 1
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"Progress: {processed}/{total_files} ({processed / total_files * 100:.2f}%) Elapsed: {minutes}:{seconds:02d}", end='\r')
    print()  # Newline after progress bar

    matrix, total_known, total_unknown = build_confusion_matrix(results, valid_label_paths, args)
    print("\nConfusion matrix (rows: predicted, cols: GT):")
    print("\t" + "\t".join(CLASS_NAMES))
    for i, row in enumerate(matrix):
        print(f"{CLASS_NAMES[i]}\t" + "\t".join(str(x) for x in row))

    known_misses = matrix[2, 0] + matrix[2, 1]
    unknown_false_alarms = matrix[0, 2] + matrix[1, 2]
    unknown_correct_rejections = matrix[2, 2]

    miss_rate = known_misses / total_known if total_known > 0 else float("nan")
    false_alarm_rate = (
        unknown_false_alarms / total_unknown if total_unknown > 0 else float("nan")
    )

    print(f"\nKnown objects: {total_known}")
    print(f"Unknown objects: {total_unknown}")
    print(f"Known miss rate: {miss_rate:.4f} ({known_misses}/{total_known})")
    print(
        f"Unknown false alarm rate: {false_alarm_rate:.4f} ({unknown_false_alarms}/{total_unknown})"
    )
    print(f"Unknown correct rejections: {unknown_correct_rejections}")

    if not args.no_save_plot:
        plot_confusion(matrix, args.save_plot)
        print(f"Saved confusion matrix plot to {args.save_plot}")


if __name__ == "__main__":
    main()

