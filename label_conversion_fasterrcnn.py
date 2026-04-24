from pathlib import Path
from typing import List, Tuple

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def yolo_xywhn_to_xyxy(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
) -> List[float]:
    """Convert normalized YOLO xywh to clamped pixel xyxy."""
    xmin = (xc - bw / 2.0) * width
    ymin = (yc - bh / 2.0) * height
    xmax = (xc + bw / 2.0) * width
    ymax = (yc + bh / 2.0) * height

    xmin = max(0.0, min(xmin, width - 1.0))
    ymin = max(0.0, min(ymin, height - 1.0))
    xmax = max(0.0, min(xmax, width - 1.0))
    ymax = max(0.0, min(ymax, height - 1.0))

    return [xmin, ymin, xmax, ymax]


def yolo_label_file_to_fasterrcnn_targets(
    label_path: Path, width: int, height: int
) -> Tuple[List[List[float]], List[int]]:
    """
    Read a YOLO txt label file and return Faster R-CNN style boxes/labels.

    Labels are shifted by +1 because torchvision reserves class 0 for background.
    """
    boxes: List[List[float]] = []
    labels: List[int] = []

    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes, labels

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            box = yolo_xywhn_to_xyxy(xc, yc, bw, bh, width, height)
            xmin, ymin, xmax, ymax = box
            if xmax <= xmin or ymax <= ymin:
                continue

            labels.append(cls_id + 1)
            boxes.append(box)

    return boxes, labels


def load_fasterrcnn_label_file(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Load precomputed Faster R-CNN labels in: class_id xmin ymin xmax ymax."""
    boxes: List[List[float]] = []
    labels: List[int] = []

    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes, labels

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(float(parts[0]))
            xmin = float(parts[1])
            ymin = float(parts[2])
            xmax = float(parts[3])
            ymax = float(parts[4])

            if xmax <= xmin or ymax <= ymin:
                continue

            labels.append(cls_id)
            boxes.append([xmin, ymin, xmax, ymax])

    return boxes, labels


def save_fasterrcnn_label_file(
    label_path: Path, boxes: List[List[float]], labels: List[int]
) -> None:
    """Save labels in Faster R-CNN text format: class_id xmin ymin xmax ymax."""
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, box in zip(labels, boxes):
            xmin, ymin, xmax, ymax = box
            f.write(f"{cls_id} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f}\n")


def convert_labels_for_images(
    images_dir: Path, src_labels_dir: Path, dst_labels_dir: Path
) -> Tuple[int, int]:
    """Convert YOLO txt labels to Faster R-CNN txt labels for one split."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Source labels directory not found: {src_labels_dir}")

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    converted = 0
    missing = 0

    for image_path in image_files:
        src_label_path = src_labels_dir / f"{image_path.stem}.txt"
        dst_label_path = dst_labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as img:
            width, height = img.size

        boxes, labels = yolo_label_file_to_fasterrcnn_targets(src_label_path, width, height)
        save_fasterrcnn_label_file(dst_label_path, boxes, labels)

        converted += 1
        if not src_label_path.exists():
            missing += 1

    return converted, missing
