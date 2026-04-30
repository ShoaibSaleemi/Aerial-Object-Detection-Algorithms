from pathlib import Path
import time
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Closed-set known classes + open-set unknown.
CLASS_NAMES = ["bird", "drone", "unknown"]

# Edit evaluation parameters here.
IMAGES_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "validation" / "labels"
IOU_THRESH = 0.5
# Pre-fusion threshold: keep candidate boxes relatively permissive so WBF can
# leverage cross-model agreement instead of dropping candidates too early.
CONF_THRESH = 0.25
FUSION_IOU_THRESH = 0.50
IMGSZ = 640
DEVICE = ""  # "cpu", "0", "0,1"; empty lets Ultralytics auto-select.

# Unknown-decision thresholds (applied after known-class weighted voting).
MIN_MODEL_SUPPORT = 2
KNOWN_FUSED_CONF_THRESH = 0.35
SCORE_MARGIN_THRESH = 0.08
DISAGREEMENT_RATIO_THRESH = 0.85

# Ensemble model list: (name, path_to_weights)
MODELS = [
    ("yolo8n", PROJECT_ROOT / "runs" / "detect" / "yolo8n" / "weights" / "best.pt"),
    ("yolo9t", PROJECT_ROOT / "runs" / "detect" / "yolo9t" / "weights" / "best.pt"),
    ("yolo10n", PROJECT_ROOT / "runs" / "detect" / "yolo10n" / "weights" / "best.pt"),
]

# Per-model per-class weighting for weighted voting / box fusion.
# We derive these from user-reported validation metrics using:
#   score = F1 * (1 - Pfa)
# then normalize per class so the strongest model for that class is 1.0.
MODEL_WEIGHTS = {
    "yolo8n": {"bird": 1.000, "drone": 1.000, "unknown": 1.000},
    "yolo9t": {"bird": 0.951, "drone": 0.944, "unknown": 0.970},
    "yolo10n": {"bird": 0.930, "drone": 0.912, "unknown": 0.948},
}

SAVE_PLOT = True
VERBOSE = False

OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "weighted_voter"
SAVE_PLOT_PATH = OUTPUT_DIR / "confusion_matrix_eval.png"
SAVE_METRICS_PLOT_PATH = OUTPUT_DIR / "metrics_table_eval.png"


def class_name_from_id(cls_id: int) -> str:
    """Map model class index to known classes, falling back to unknown."""
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
    """Build class-id -> {bird, drone, unknown} map from model.names."""
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
    """Greedy IoU clustering across all detections regardless of class."""
    clusters = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        matched = False
        for cluster in clusters:
            # Compare with current cluster representative box.
            if compute_iou(det["box"], cluster["rep_box"]) >= iou_thresh:
                cluster["items"].append(det)
                # Representative box is mean of member boxes for stable growth.
                boxes = np.array([item["box"] for item in cluster["items"]], dtype=np.float32)
                cluster["rep_box"] = boxes.mean(axis=0).tolist()
                matched = True
                break
        if not matched:
            clusters.append({"items": [det], "rep_box": det["box"][:]})
    return [c["items"] for c in clusters]


def fuse_cluster(cluster_items):
    """Fuse one overlap cluster using requested WBF equations.

    Standard WBF confidence:
        Confidence_fused = (C1 + C2 + ... + CT) / T

    Weighted WBF box coordinates:
        Weighted_score_i = model_weight_i * confidence_i
        X1_fused = sum(Weighted_score_i * X1_i) / sum(Weighted_score_i)
        (same for Y1, X2, Y2)

    Class vote:
        Class_score(class) = sum(model_weight_i * confidence_i)
        Final_class = argmax(Class_score(class))
    """
    # Weighted class vote across known classes.
    class_scores = {0: 0.0, 1: 0.0}
    for det in cluster_items:
        if det["class_id"] in (0, 1):
            class_scores[det["class_id"]] += det["weighted_score"]

    # Find best and second known scores to measure uncertainty.
    best_known_class = max(class_scores.items(), key=lambda kv: kv[1])[0]
    second_known_class = 1 - best_known_class
    best_score = class_scores[best_known_class]
    second_score = class_scores[second_known_class]
    score_margin = best_score - second_score
    disagreement_ratio = second_score / (best_score + 1e-12)

    # Count how many distinct models support the best known class.
    best_known_support_models = {
        det["model"]
        for det in cluster_items
        if det["class_id"] == best_known_class
    }
    support_count = len(best_known_support_models)

    # Compute best-known fused confidence (paper mean-confidence definition)
    # and use it in the unknown decision logic.
    best_known_items = [d for d in cluster_items if d["class_id"] == best_known_class]
    best_known_fused_conf = (
        float(np.mean([d["confidence"] for d in best_known_items]))
        if best_known_items
        else 0.0
    )

    # Unknown decision layer after weighted voting:
    # if support/confidence/margin/disagreement indicate uncertainty,
    # classify as unknown instead of forcing bird/drone.
    uncertain = (
        support_count < MIN_MODEL_SUPPORT
        or best_known_fused_conf < KNOWN_FUSED_CONF_THRESH
        or score_margin < SCORE_MARGIN_THRESH
        or disagreement_ratio > DISAGREEMENT_RATIO_THRESH
    )

    if uncertain:
        final_class = 2
        # For unknown, keep cluster geometry by fusing all members.
        chosen = list(cluster_items)
    else:
        final_class = best_known_class
        chosen = best_known_items

    if not chosen:
        return None

    # Standard WBF fused confidence (mean confidence over selected members).
    fused_conf = float(np.mean([d["confidence"] for d in chosen]))

    # Weighted WBF fused coordinates using (model_weight * confidence).
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
        "support_count": int(support_count),
        "score_margin": float(score_margin),
        "disagreement_ratio": float(disagreement_ratio),
    }


def run_weighted_boxes_fusion_on_image(models, image_path: Path):
    """Run all models once on one image and return both raw model and fused outputs.

    Returns:
        fused: list of fused detections for WBF
        per_model_predictions: dict[model_name] -> list of detections in the same
            minimal format used by evaluation ({box, class_id, confidence}).
    """
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
            weighted_score = float(model_weight * conf)

            # Cache each model's raw prediction once so we can evaluate model-level
            # metrics later without running an extra inference pass.
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
                    "model_weight": float(model_weight),
                    "weighted_score": weighted_score,
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


def build_all_confusion_matrices(predictions_by_method, label_paths, images_dir, iou_thresh, verbose):
    """Build confusion matrices for all methods in one pass over labels/images.

    This avoids re-reading labels and repeating image-size parsing per method,
    which keeps multi-method reporting efficient.
    """
    matrices = {method_name: np.zeros((3, 3), dtype=int) for method_name in predictions_by_method}
    missed_by_method = {
        method_name: np.zeros((3,), dtype=int) for method_name in predictions_by_method
    }
    unmatched_pred_by_method = {
        method_name: np.zeros((3,), dtype=int) for method_name in predictions_by_method
    }
    total_images = len(label_paths)
    start_time = time.time()

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

        for method_name, method_preds_per_image in predictions_by_method.items():
            preds = method_preds_per_image[image_idx]
            pred_boxes = [p["box"] for p in preds]
            pred_labels = [p["class_id"] for p in preds]

            assignments, used_pred = match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh)

            for gt_idx, gt_label in enumerate(gt_labels):
                if gt_idx in assignments:
                    pred_idx = assignments[gt_idx]
                    pred_label = pred_labels[pred_idx]
                    matrices[method_name][pred_label, gt_label] += 1
                else:
                    # Missed detection is tracked separately from predicted unknown.
                    missed_by_method[method_name][gt_label] += 1

            # Unmatched predictions are false positives that should not be treated
            # as matched unknown decisions.
            for pred_idx, pred_label in enumerate(pred_labels):
                if pred_idx not in used_pred:
                    unmatched_pred_by_method[method_name][pred_label] += 1

        if verbose and len(gt_labels) > 0:
            print(
                f"{image_name}: GT {len(gt_labels)}"
            )

        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Matrix progress: {image_idx + 1}/{total_images} ({(image_idx + 1) / total_images * 100:.2f}%) "
            f"Elapsed: {minutes}:{seconds:02d}",
            end="\r",
        )

    print()
    return matrices, missed_by_method, unmatched_pred_by_method


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


def compute_metrics_from_confusion(matrix, missed_counts=None, unmatched_pred_counts=None):
    """
    rows = predicted, cols = ground truth
    """
    if missed_counts is None:
        missed_counts = np.zeros((3,), dtype=int)
    if unmatched_pred_counts is None:
        unmatched_pred_counts = np.zeros((3,), dtype=int)

    n_classes = matrix.shape[0]
    total_gt = int(matrix.sum() + np.sum(missed_counts))

    per_class_metrics = []

    for c in range(n_classes):
        tp = int(matrix[c, c])
        fp = int(matrix[c, :].sum() - tp + unmatched_pred_counts[c])
        fn = int(matrix[:, c].sum() - tp + missed_counts[c])
        tn = int(max(total_gt - tp - fp - fn, 0))

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

    total_known = int(matrix[:, 0].sum() + matrix[:, 1].sum() + missed_counts[0] + missed_counts[1])
    total_unknown = int(matrix[:, 2].sum() + missed_counts[2])

    known_misses = int(missed_counts[0] + missed_counts[1])
    unknown_false_alarms = int(matrix[0, 2] + matrix[1, 2])
    unknown_correct_rejections = int(matrix[2, 2])

    summary_metrics = {
        "Known objects": total_known,
        "Unknown objects": total_unknown,
        "Missed known objects": known_misses,
        "Missed unknown objects": int(missed_counts[2]),
        "Unmatched predicted bird": int(unmatched_pred_counts[0]),
        "Unmatched predicted drone": int(unmatched_pred_counts[1]),
        "Unmatched predicted unknown": int(unmatched_pred_counts[2]),
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
    print(f"Missed known objects: {summary_metrics['Missed known objects']}")
    print(f"Missed unknown objects: {summary_metrics['Missed unknown objects']}")
    print(f"Known miss rate: {fmt_pct(summary_metrics['Known miss rate'])}")
    print(f"Unknown false alarm rate: {fmt_pct(summary_metrics['Unknown false alarm rate'])}")
    print(f"Unknown correct rejections: {summary_metrics['Unknown correct rejections']}")
    print(f"Unmatched predicted bird: {summary_metrics['Unmatched predicted bird']}")
    print(f"Unmatched predicted drone: {summary_metrics['Unmatched predicted drone']}")
    print(f"Unmatched predicted unknown: {summary_metrics['Unmatched predicted unknown']}")


def print_method_metrics_summary_table(metrics_by_method, method_order):
    """Print one compact table for per-model metrics and WBF together.

    Each row is a method (single model or ensemble), so this acts as a quick
    side-by-side comparison without printing per-model confusion matrices.
    """
    print("\nMethod comparison (metrics only):")
    header = (
        f"{'Method':<22}"
        f"{'Macro Prec':>12}{'Macro Recall':>14}{'Macro F1':>12}"
        f"{'Macro Pfa':>12}{'Macro P(success)':>18}"
        f"{'Unknown FAR':>14}{'Open-set P(success)':>20}"
    )
    print(header)
    print("-" * len(header))

    for method_name in method_order:
        per_class_metrics, macro_metrics, summary_metrics = metrics_by_method[method_name]
        open_set_p_success = (
            1.0 - summary_metrics["Known miss rate"]
            if not np.isnan(summary_metrics["Known miss rate"])
            else float("nan")
        )
        print(
            f"{method_name:<22}"
            f"{fmt_pct(macro_metrics['Precision']):>12}"
            f"{fmt_pct(macro_metrics['Recall']):>14}"
            f"{fmt_pct(macro_metrics['F1-score']):>12}"
            f"{fmt_pct(macro_metrics['False Positive Rate']):>12}"
            f"{fmt_pct(macro_metrics['Detection Probability']):>18}"
            f"{fmt_pct(summary_metrics['Unknown false alarm rate']):>14}"
            f"{fmt_pct(open_set_p_success):>20}"
        )


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
        f"Missed known: {summary_metrics['Missed known objects']}    "
        f"Missed unknown: {summary_metrics['Missed unknown objects']}    "
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ensemble models from trained run outputs.
    loaded_models = []
    print("Loading ensemble models...")
    for model_name, model_path in MODELS:
        if not model_path.exists():
            print(f"[SKIP] {model_name}: file not found at {model_path}")
            continue
        model = YOLO(str(model_path))
        class_map = build_model_class_map(model)
        print(f"[INFO] {model_name} model.names = {model.names}")
        loaded_models.append((model_name, model, class_map))
        print(f"[OK]   {model_name}: {model_path.name}")

    if len(loaded_models) == 0:
        raise ValueError("No ensemble models were loaded. Check MODELS paths.")

    label_dir = LABELS_DIR
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = list(label_dir.glob("*.txt"))
    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

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
            image_paths.append(image_path)
            valid_label_paths.append(label_path)

    if len(image_paths) == 0:
        raise ValueError(f"No validation images found in {IMAGES_DIR}")

    print(f"\nRunning weighted box fusion inference on {len(image_paths)} validation images...")

    fused_results_per_image = []
    per_model_results = {model_name: [] for model_name, _, _ in loaded_models}
    total_files = len(image_paths)
    start_time = time.time()

    for i, image_path in enumerate(image_paths, 1):
        fused, per_model_preds = run_weighted_boxes_fusion_on_image(loaded_models, image_path)
        fused_results_per_image.append(fused)
        for model_name in per_model_results:
            per_model_results[model_name].append(per_model_preds[model_name])

        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(
            f"Inference progress: {i}/{total_files} ({i / total_files * 100:.2f}%) Elapsed: {minutes}:{seconds:02d}",
            end="\r",
        )
    print()

    predictions_by_method = {**per_model_results, "WBF ensemble": fused_results_per_image}
    matrices, missed_by_method, unmatched_pred_by_method = build_all_confusion_matrices(
        predictions_by_method,
        valid_label_paths,
        IMAGES_DIR,
        IOU_THRESH,
        VERBOSE,
    )

    metrics_by_method = {
        method_name: compute_metrics_from_confusion(
            matrix,
            missed_counts=missed_by_method[method_name],
            unmatched_pred_counts=unmatched_pred_by_method[method_name],
        )
        for method_name, matrix in matrices.items()
    }

    method_order = [model_name for model_name, _, _ in loaded_models] + ["WBF ensemble"]
    print_method_metrics_summary_table(metrics_by_method, method_order)

    matrix = matrices["WBF ensemble"]

    print("\nConfusion matrix (rows: predicted, cols: GT):")
    print("\t" + "\t".join(CLASS_NAMES))
    for i, row in enumerate(matrix):
        print(f"{CLASS_NAMES[i]}\t" + "\t".join(str(x) for x in row))

    per_class_metrics, macro_metrics, summary_metrics = metrics_by_method["WBF ensemble"]
    print_metrics_table(per_class_metrics, macro_metrics, summary_metrics)
    print()

    if SAVE_PLOT:
        plot_confusion(matrix, SAVE_PLOT_PATH, "Weighted Boxes Fusion")
        print(f"Saved confusion matrix plot to {SAVE_PLOT_PATH}")

        plot_metrics_table(
            per_class_metrics=per_class_metrics,
            macro_metrics=macro_metrics,
            summary_metrics=summary_metrics,
            save_path=SAVE_METRICS_PLOT_PATH,
            title_prefix="Weighted Boxes Fusion",
        )
        print(f"Saved metrics table plot to {SAVE_METRICS_PLOT_PATH}")


if __name__ == "__main__":
    main()
