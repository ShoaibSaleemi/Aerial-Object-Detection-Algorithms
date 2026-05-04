import csv
from pathlib import Path
import random
import time
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

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
FUSION_IOU_THRESH = 0.50
IMGSZ = 640
DEVICE = ""  # "cpu", "0", "0,1"; empty lets Ultralytics auto-select.

# Unknown-decision thresholds (applied after known-class weighted voting).
MIN_MODEL_SUPPORT = 5
KNOWN_FUSED_CONF_THRESH = 0.5601453198085411
SCORE_MARGIN_THRESH = 0.09125076880326494
DISAGREEMENT_RATIO_THRESH = 0.15533651974598006

# Per-model per-class weighting for weighted voting / box fusion.
MODEL_WEIGHTS = {
    "yolo8n":  {"bird": 1.0814851758318496,  "drone": 1.1806397062045533,  "unknown": 1.5854956737557213},
    "yolo9t":  {"bird": 1.1648048860139104,  "drone": 1.9507533272268047,  "unknown": 1.5176441781892689},
    "yolo10n": {"bird": 1.012484828994974,   "drone": 1.238554113695587,   "unknown": 0.9796306541223246},
    "yolo11n": {"bird": 1.2533830034169342,  "drone": 1.1708816661545447,  "unknown": 0.9072305499609367},
    "yolo12n": {"bird": 1.2217564703689683,  "drone": 1.703242413959964,   "unknown": 1.5846263992743486},
    "yolo26n": {"bird": 0.7438992484903112,  "drone": 1.0219630042404741,  "unknown": 1.585134798429956},
}

# Ensemble model list: (name, path_to_weights)
MODELS = [
    ("yolo8n", PROJECT_ROOT / "runs" / "detect" / "yolo8n" / "weights" / "best.pt"),
    ("yolo9t", PROJECT_ROOT / "runs" / "detect" / "yolo9t" / "weights" / "best.pt"),
    ("yolo10n", PROJECT_ROOT / "runs" / "detect" / "yolo10n" / "weights" / "best.pt"),
    ("yolo11n", PROJECT_ROOT / "runs" / "detect" / "yolo11n" / "weights" / "best.pt"),
    ("yolo12n", PROJECT_ROOT / "runs" / "detect" / "yolo12n" / "weights" / "best.pt"),
    ("yolo26n", PROJECT_ROOT / "runs" / "detect" / "yolo26n" / "weights" / "best.pt"),
]

SAVE_PLOT = True
VERBOSE = False

OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "weighted_voter"
SAVE_PLOT_PATH = OUTPUT_DIR / "confusion_matrix_eval.png"
SAVE_METRICS_CSV_PATH = OUTPUT_DIR / "metrics_table_eval.csv"


def class_name_from_id(cls_id: int) -> str:
    if cls_id == 0:
        return "bird"
    if cls_id == 1:
        return "drone"
    return "unknown"


def normalize_class_name(name: str) -> str:
    lower = str(name).strip().lower()
    if "bird" in lower:
        return "bird"
    if "drone" in lower:
        return "drone"
    return "unknown"


def build_model_class_map(model):
    names = getattr(model, "names", None)
    if names is None:
        return {}

    if isinstance(names, dict):
        return {int(k): normalize_class_name(v) for k, v in names.items()}

    if isinstance(names, list):
        return {i: normalize_class_name(v) for i, v in enumerate(names)}

    return {}


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


def cluster_detections(detections, iou_thresh):
    clusters = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        matched = False
        for cluster in clusters:
            if compute_iou(det["box"], cluster["rep_box"]) >= iou_thresh:
                cluster["items"].append(det)
                boxes = np.array([item["box"] for item in cluster["items"]], dtype=np.float32)
                cluster["rep_box"] = boxes.mean(axis=0).tolist()
                matched = True
                break
        if not matched:
            clusters.append({"items": [det], "rep_box": det["box"][:]})
    return [c["items"] for c in clusters]


def fuse_cluster(cluster_items):
    class_scores = {0: 0.0, 1: 0.0}
    for det in cluster_items:
        if det["class_id"] in (0, 1):
            class_scores[det["class_id"]] += det["weighted_score"]

    best_known_class = max(class_scores.items(), key=lambda kv: kv[1])[0]
    second_known_class = 1 - best_known_class
    best_score = class_scores[best_known_class]
    second_score = class_scores[second_known_class]
    score_margin = best_score - second_score
    disagreement_ratio = second_score / (best_score + 1e-12)

    best_known_support_models = {
        det["model"]
        for det in cluster_items
        if det["class_id"] == best_known_class
    }
    support_count = len(best_known_support_models)

    best_known_items = [d for d in cluster_items if d["class_id"] == best_known_class]
    best_known_fused_conf = (
        float(np.mean([d["confidence"] for d in best_known_items]))
        if best_known_items
        else 0.0
    )

    uncertain = (
        support_count < MIN_MODEL_SUPPORT
        or best_known_fused_conf < KNOWN_FUSED_CONF_THRESH
        or score_margin < SCORE_MARGIN_THRESH
        or disagreement_ratio > DISAGREEMENT_RATIO_THRESH
    )

    if uncertain:
        final_class = 2
        chosen = list(cluster_items)
    else:
        final_class = best_known_class
        chosen = best_known_items

    if not chosen:
        return None

    fused_conf = float(np.mean([d["confidence"] for d in chosen]))

    score_sum = sum(d["weighted_score"] for d in chosen)
    if score_sum <= 0:
        return None

    x1 = sum(d["weighted_score"] * d["box"][0] for d in chosen) / score_sum
    y1 = sum(d["weighted_score"] * d["box"][1] for d in chosen) / score_sum
    x2 = sum(d["weighted_score"] * d["box"][2] for d in chosen) / score_sum
    y2 = sum(d["weighted_score"] * d["box"][3] for d in chosen) / score_sum

    return {
        "box": [float(x1), float(y1), float(x2), float(y2)],
        "class_id": int(final_class),
        "confidence": fused_conf,
    }


def run_weighted_boxes_fusion_on_image(models, image_path: Path):
    all_detections = []
    per_model_predictions = {model_name: [] for model_name, _, _ in models}

    for model_name, model, class_map in models:
        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESH,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
        )
        result = results[0] if isinstance(results, list) else results

        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy().astype(float)

        for box, cls_id, conf in zip(boxes_xyxy, class_ids, confidences):
            cls_name = class_map.get(int(cls_id), class_name_from_id(int(cls_id)))
            cls_idx = 0 if cls_name == "bird" else 1 if cls_name == "drone" else 2
            model_weight = MODEL_WEIGHTS.get(model_name, {}).get(cls_name, 1.0)

            per_model_predictions[model_name].append(
                {
                    "box": [float(v) for v in box.tolist()],
                    "class_id": cls_idx,
                    "confidence": float(conf),
                }
            )

            all_detections.append(
                {
                    "box": [float(v) for v in box.tolist()],
                    "class_id": cls_idx,
                    "confidence": float(conf),
                    "model": model_name,
                    "weighted_score": float(model_weight * conf),
                }
            )

    if not all_detections:
        return [], per_model_predictions

    clusters = cluster_detections(all_detections, FUSION_IOU_THRESH)
    fused = []
    for cluster in clusters:
        item = fuse_cluster(cluster)
        if item is not None:
            fused.append(item)

    return fused, per_model_predictions


def build_confusion_matrix(fused_results, label_paths, images_dir, iou_thresh, verbose):
    matrix = np.zeros((3, 3), dtype=int)
    total_known = 0
    total_unknown = 0
    total_images = len(label_paths)

    for image_idx, label_path in enumerate(sorted(label_paths)):
        image_name = label_path.stem
        image_path = Path(images_dir) / f"{image_name}.jpg"
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

        preds = fused_results[image_idx]
        pred_boxes = [p["box"] for p in preds]
        pred_labels = [p["class_id"] for p in preds]

        assignments, _ = match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh)

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

    return matrix, total_known, total_unknown


def plot_confusion(matrix, save_path, title_prefix):
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
    orange = "\033[38;5;214m"
    reset = "\033[0m"
    lines.append(
        f"{'macro-avg':<10}"
        + orange
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_models = []
    print("Loading ensemble models...")
    for model_name, model_path in MODELS:
        if not model_path.exists():
            print(f"[SKIP] {model_name}: file not found at {model_path}")
            continue
        model = YOLO(str(model_path))
        class_map = build_model_class_map(model)
        loaded_models.append((model_name, model, class_map))
        print(f"[OK]   {model_name}: {model_path.name}")

    if len(loaded_models) == 0:
        raise ValueError("No ensemble models were loaded. Check MODELS paths.")

    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABELS_DIR}")

    label_paths = sorted(list(LABELS_DIR.glob("*.txt")))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {LABELS_DIR}")

    image_paths = []
    valid_label_paths = []
    for label_path in label_paths:
        image_name = label_path.stem
        image_path = IMAGES_DIR / f"{image_name}.jpg"
        if not image_path.exists():
            for ext in [".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = IMAGES_DIR / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path.exists():
            image_paths.append(image_path)
            valid_label_paths.append(label_path)

    if len(image_paths) == 0:
        raise ValueError(f"No validation images found in {IMAGES_DIR}")

    print(f"Running weighted box fusion inference on {len(image_paths)} validation images...")
    total_files = len(image_paths)
    start_time = time.time()

    fused_results_per_image = []
    per_model_results = {model_name: [] for model_name, _, _ in loaded_models}
    for idx, image_path in enumerate(image_paths, 1):
        fused, per_model_preds = run_weighted_boxes_fusion_on_image(loaded_models, image_path)
        fused_results_per_image.append(fused)
        for model_name in per_model_results:
            per_model_results[model_name].append(per_model_preds[model_name])

        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Progress: {idx}/{total_files} ({idx / total_files * 100:.2f}%) \033[38;5;214mElapsed: {minutes}:{seconds:02d}\033[0m",
            end="\r",
        )
    print()

    # Print per-model metrics first.
    for model_name, _, _ in loaded_models:
        model_matrix, _, _ = build_confusion_matrix(
            per_model_results[model_name],
            valid_label_paths,
            IMAGES_DIR,
            IOU_THRESH,
            VERBOSE,
        )
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")
        pcm, mm, sm = compute_metrics_from_confusion(model_matrix)
        print_confusion_and_metrics_side_by_side(model_matrix, pcm, mm)

    matrix, total_known, total_unknown = build_confusion_matrix(
        fused_results_per_image,
        valid_label_paths,
        IMAGES_DIR,
        IOU_THRESH,
        VERBOSE,
    )

    print(f"\n{'=' * 70}")
    print("Ensemble: Weighted Boxes Fusion")
    print(f"{'=' * 70}")
    per_class_metrics, macro_metrics, summary_metrics = compute_metrics_from_confusion(matrix)
    print_confusion_and_metrics_side_by_side(matrix, per_class_metrics, macro_metrics)
    print()

    if SAVE_PLOT:
        plot_confusion(matrix, SAVE_PLOT_PATH, "Weighted Boxes Fusion")
        print(f"Saved confusion matrix plot to {SAVE_PLOT_PATH}")

        save_metrics_table_csv(
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            save_path=SAVE_METRICS_CSV_PATH,
        )
        print(f"Saved metrics table CSV to {SAVE_METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
