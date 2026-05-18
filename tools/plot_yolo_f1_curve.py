"""
Plot F1 vs confidence threshold curves for all discovered YOLO models.

Reads pre-computed prediction caches (pred_cache_*_imgsz640.json) from
runs/detect/weights/ and sweeps confidence thresholds to compute F1.
Best operating points are loaded from best_*.json and marked with a star (*).

Usage:
    python tools/plot_yolo_f1_curve.py
    python tools/plot_yolo_f1_curve.py --steps 200 --min-conf 0.05 --max-conf 0.75
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "runs" / "detect" / "weights"

IOU_THRESH = 0.5
IMGSZ = 640

MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n"]

COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # yellow-green
    "#7f7f7f",  # gray
]


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def compute_iou(box_a: list, box_b: list) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(gt_boxes: list, pred_boxes: list, iou_thresh: float) -> dict:
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return {}
    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(n_pred):
            ious[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])
    assignments: dict = {}
    while True:
        max_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_idx] < iou_thresh:
            break
        gt_idx, pred_idx = max_idx
        assignments[gt_idx] = pred_idx
        ious[gt_idx, :] = -1.0
        ious[:, pred_idx] = -1.0
    return assignments


def compute_f1_at_conf(cached_data: list, conf_thresh: float) -> float:
    matrix = np.zeros((3, 3), dtype=int)
    for entry in cached_data:
        gt_boxes = entry["gt_boxes"]
        gt_labels = entry["gt_labels"]
        preds = [(c, cls, box) for c, cls, box in entry["preds"] if c >= conf_thresh]
        pred_boxes = [p[2] for p in preds]
        pred_labels = [p[1] for p in preds]
        assignments = match_predictions(gt_boxes, pred_boxes, IOU_THRESH)
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx in assignments:
                matrix[pred_labels[assignments[gt_idx]], gt_label] += 1
            else:
                matrix[2, gt_label] += 1

    f1s = []
    for c in range(3):
        tp = int(matrix[c, c])
        fp = int(matrix[c, :].sum() - tp)
        fn = int(matrix[:, c].sum() - tp)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        if np.isnan(prec) or np.isnan(rec):
            f1s.append(float("nan"))
        else:
            f1s.append(safe_div(2 * prec * rec, prec + rec))
    return float(np.nanmean(f1s))


def discover_models() -> list[str]:
    cache_files = sorted(WEIGHTS_DIR.glob(f"pred_cache_*_imgsz{IMGSZ}.json"))
    names = []
    for cf in cache_files:
        name = cf.stem.replace("pred_cache_", "").rsplit(f"_imgsz{IMGSZ}", 1)[0]
        names.append(name)

    def sort_key(n: str) -> int:
        try:
            return MODEL_ORDER.index(n)
        except ValueError:
            return len(MODEL_ORDER)

    names.sort(key=sort_key)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot F1 vs confidence threshold for all YOLO models")
    parser.add_argument("--steps", type=int, default=200, help="Number of confidence threshold steps to sweep")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Lower bound for confidence range")
    parser.add_argument("--max-conf", type=float, default=0.75, help="Upper bound for confidence range")
    args = parser.parse_args()

    conf_values = np.linspace(args.min_conf, args.max_conf, args.steps)

    model_names = discover_models()
    if not model_names:
        raise FileNotFoundError(f"No pred_cache files found in {WEIGHTS_DIR}")

    print(f"Found {len(model_names)} model(s): {', '.join(model_names)}")
    print(f"Sweeping conf [{args.min_conf:.2f}, {args.max_conf:.2f}] in {args.steps} steps\n")

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model_name in enumerate(model_names):
        cache_path = WEIGHTS_DIR / f"pred_cache_{model_name}_imgsz{IMGSZ}.json"
        print(f"  [{i + 1}/{len(model_names)}] Computing F1 curve for {model_name}...", flush=True)

        with cache_path.open("r", encoding="utf-8") as f:
            cached_data = json.load(f)

        f1_values = [compute_f1_at_conf(cached_data, float(c)) for c in conf_values]

        color = COLORS[i % len(COLORS)]
        ax.plot(conf_values, f1_values, color=color, linewidth=1.8, label=model_name)

        # Load best conf/F1 from JSON snapshot if available, else find from sweep
        best_json_path = WEIGHTS_DIR / f"best_{model_name}.json"
        if best_json_path.exists():
            with best_json_path.open("r", encoding="utf-8") as f:
                best_data = json.load(f)
            best_conf = float(best_data["best"]["conf_thresh"])
            best_f1 = float(best_data["best"]["f1"])
        else:
            best_idx = int(np.nanargmax(f1_values))
            best_conf = float(conf_values[best_idx])
            best_f1 = float(f1_values[best_idx])

        ax.plot(
            best_conf, best_f1,
            marker="*", markersize=14,
            color=color,
            markeredgecolor="black", markeredgewidth=0.5,
            zorder=5, linestyle="None",
        )
        ax.annotate(
            f"  {best_f1:.4f}",
            xy=(best_conf, best_f1),
            fontsize=7.5,
            color=color,
            va="center",
        )

    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("F1 Score vs Confidence Threshold — YOLO Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(args.min_conf, args.max_conf)

    out_path = WEIGHTS_DIR / "f1_vs_conf_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
