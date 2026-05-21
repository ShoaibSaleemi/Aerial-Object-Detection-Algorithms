"""
PR-curve plotter for YOLO models in runs/detect/weights/.

Auto-discovers .pt weight files, runs a single low-confidence inference pass
on the dataset 2 test split (with per-model pred-cache), sweeps confidence
thresholds to build a macro-averaged PR curve, and saves PNG + PDF.

Caches
------
  pred_cache_<model>_dataset2_imgsz640.json   — raw predictions (one GPU pass)
  sweep_cache_<model>_<split>_steps<n>_...npz — macro P/R arrays

Usage
-----
    python tools/plot_yolo_pr_curves_weights.py                     # all models, test split
    python tools/plot_yolo_pr_curves_weights.py --model yolo8n      # one model only
    python tools/plot_yolo_pr_curves_weights.py --split test        # split label in filename
    python tools/plot_yolo_pr_curves_weights.py --steps 2000        # sweep steps
    python tools/plot_yolo_pr_curves_weights.py --out-dir runs/detect/weights/plots
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
WEIGHTS_DIR     = PROJECT_ROOT / "runs" / "detect" / "weights"
DATASET_DIR     = PROJECT_ROOT / "dataset 2"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
TEST_LABELS_DIR = DATASET_DIR / "test" / "labels"
DATASET_TAG     = DATASET_DIR.name.replace(" ", "")   # "dataset2"

# ---------------------------------------------------------------------------
# Detection constants
# ---------------------------------------------------------------------------

CLASS_NAMES      = ["bird", "drone", "unknown"]
N_CLASSES        = len(CLASS_NAMES)
IOU_THRESH       = 0.5
IMGSZ            = 640
CONF_INFER       = 0.001
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# ---------------------------------------------------------------------------
# Per-model confidence thresholds  (star marker on the PR curve)
# ---------------------------------------------------------------------------

MODEL_CONF_THRESH = {
    "yolo8n":  0.6863484706628682,
    "yolo8m":  0.7133918823950539,
    "yolo9t":  0.6724046133517759,
    "yolo10n": 0.5910035105688879,
    "yolo11n": 0.712997868833143,
    "yolo12n": 0.6838702654977842,
    "yolo26n": 0.6052508580184951,
}

MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n"]

# ---------------------------------------------------------------------------
# Plot style — edit these to change appearance without touching the logic
# ---------------------------------------------------------------------------

PLOT_FIGSIZE          = (7, 6)
PLOT_LINEWIDTH        = 2.5
PLOT_ALPHA            = 1.0
PLOT_COLORMAP         = "plasma"
PLOT_SMOOTH_SIGMA     = 9.0       # Gaussian smoothing sigma (0 = off)

PLOT_XLIM             = (0.0, 1.0)
PLOT_YLIM             = (0.0, 1.05)

PLOT_XLABEL_FONTSIZE  = 20
PLOT_YLABEL_FONTSIZE  = 20
PLOT_TITLE_FONTSIZE   = 20
PLOT_TICK_FONTSIZE    = 20
PLOT_LEGEND_FONTSIZE  = 20

PLOT_GRID_ALPHA       = 0.3
PLOT_GRID_LINESTYLE   = "--"
PLOT_GRID_LINEWIDTH   = 0.7

PLOT_MARKER           = "$*$"     # best-point marker
PLOT_MARKER_SIZE      = 20
PLOT_MARKER_COLOR     = "black"

PLOT_DPI              = 150

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_STATUS_LEN = 0


def _status(msg: str) -> None:
    global _STATUS_LEN
    if sys.stdout.isatty():
        print(f"\r{msg.ljust(_STATUS_LEN)}", end="", flush=True)
        _STATUS_LEN = len(msg)
    else:
        print(msg, flush=True)


def _status_end() -> None:
    global _STATUS_LEN
    if sys.stdout.isatty():
        print()
    _STATUS_LEN = 0


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    radius = max(1, int(3 * sigma))
    x      = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    valid = ~(np.isnan(recall) | np.isnan(precision))
    r, p  = recall[valid], precision[valid]
    if len(r) < 2:
        return float("nan")
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def xywhn_to_xyxy(box: tuple, w: int, h: int) -> list:
    cx, cy, bw, bh = box
    return [
        (cx - bw / 2.0) * w,
        (cy - bh / 2.0) * h,
        (cx + bw / 2.0) * w,
        (cy + bh / 2.0) * h,
    ]


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorised IoU computation: gt (n,4), pred (m,4) -> (n,m)."""
    gt_e   = gt[:, np.newaxis, :]
    pred_e = pred[np.newaxis, :, :]
    x1 = np.maximum(gt_e[..., 0], pred_e[..., 0])
    y1 = np.maximum(gt_e[..., 1], pred_e[..., 1])
    x2 = np.minimum(gt_e[..., 2], pred_e[..., 2])
    y2 = np.minimum(gt_e[..., 3], pred_e[..., 3])
    inter     = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_gt   = (np.maximum(0.0, gt_e[..., 2] - gt_e[..., 0])
                 * np.maximum(0.0, gt_e[..., 3] - gt_e[..., 1]))
    area_pred = (np.maximum(0.0, pred_e[..., 2] - pred_e[..., 0])
                 * np.maximum(0.0, pred_e[..., 3] - pred_e[..., 1]))
    union = area_gt + area_pred - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(iou_sub: np.ndarray) -> dict:
    """Greedy IoU-based matching. Works on a copy to avoid mutating caller data."""
    assignments: dict = {}
    iou = iou_sub.copy()
    while True:
        idx = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[idx] < IOU_THRESH:
            break
        gi, pi = int(idx[0]), int(idx[1])
        assignments[gi] = pi
        iou[gi, :] = -1.0
        iou[:, pi] = -1.0
    return assignments


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_label_file(path: Path) -> tuple[list, list]:
    boxes: list  = []
    labels: list = []
    if not path.exists():
        return boxes, labels
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            boxes.append(tuple(map(float, parts[1:])))
            labels.append(cls if cls in (0, 1) else 2)
    return boxes, labels


def find_test_pairs() -> tuple[list, list]:
    image_paths: list = []
    label_paths: list = []
    for lp in sorted(TEST_LABELS_DIR.glob("*.txt")):
        for ext in IMAGE_EXTENSIONS:
            ip = TEST_IMAGES_DIR / f"{lp.stem}{ext}"
            if ip.exists():
                image_paths.append(ip)
                label_paths.append(lp)
                break
    if not image_paths:
        raise FileNotFoundError(
            f"No test image-label pairs found.\n"
            f"  Images: {TEST_IMAGES_DIR}\n"
            f"  Labels: {TEST_LABELS_DIR}"
        )
    return image_paths, label_paths


# ---------------------------------------------------------------------------
# Prediction caching (one GPU pass per model)
# ---------------------------------------------------------------------------

def _build_pred_cache(
    model_path: Path,
    image_paths: list,
    label_paths: list,
    cache_path: Path,
) -> list:
    print(f"  Loading model weights: {model_path.name}")
    model = YOLO(str(model_path))
    total = len(image_paths)
    cached: list = []
    t0 = time.time()

    for idx, (ip, lp) in enumerate(zip(image_paths, label_paths), 1):
        gt_boxes_xywh, gt_labels = load_label_file(lp)
        with Image.open(ip) as img:
            width, height = img.size
        gt_boxes_xyxy = [xywhn_to_xyxy(b, width, height) for b in gt_boxes_xywh]

        result = model.predict(source=str(ip), conf=CONF_INFER, imgsz=IMGSZ, verbose=False)
        result = result[0] if isinstance(result, list) else result

        preds: list = []
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                cls_id = int(cls)
                preds.append(
                    (float(conf), cls_id if cls_id in (0, 1) else 2, [float(v) for v in box])
                )

        cached.append({"gt_boxes": gt_boxes_xyxy, "gt_labels": gt_labels, "preds": preds})

        elapsed = time.time() - t0
        h, rem = divmod(int(elapsed), 3600)
        m, s   = divmod(rem, 60)
        _status(
            f"  Caching {idx}/{total} ({idx / total * 100:.1f}%)"
            f"  |  elapsed {h}:{m:02d}:{s:02d}"
        )

    _status_end()
    del model
    torch.cuda.empty_cache()

    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cached, f)
    tmp.replace(cache_path)
    print(f"  Saved pred cache: {cache_path.name}  ({total} images)")
    return cached


def load_or_create_pred_cache(
    model_name: str,
    model_path: Path,
    image_paths: list,
    label_paths: list,
) -> list:
    cache_path = WEIGHTS_DIR / f"pred_cache_{model_name}_{DATASET_TAG}_imgsz{IMGSZ}.json"
    if cache_path.exists():
        try:
            print(f"  Loading pred cache: {cache_path.name}")
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Cache ready  ({len(data)} images)")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Cache corrupt ({exc}) — re-running inference")
            cache_path.unlink(missing_ok=True)
    return _build_pred_cache(model_path, image_paths, label_paths, cache_path)


# ---------------------------------------------------------------------------
# IoU precomputation (one-time per model, before the conf sweep)
# ---------------------------------------------------------------------------

def precompute_iou_matrices(cached_data: list) -> list:
    precomputed: list = []
    for entry in cached_data:
        gt_labels    = entry["gt_labels"]
        n_gt         = len(gt_labels)
        preds_sorted = sorted(entry["preds"], key=lambda x: x[0])
        pred_confs   = np.array([p[0] for p in preds_sorted], dtype=np.float32)
        pred_labels  = [p[1] for p in preds_sorted]

        if n_gt > 0 and len(preds_sorted) > 0:
            gt_arr   = np.array(entry["gt_boxes"], dtype=np.float32)
            pred_arr = np.array([p[2] for p in preds_sorted], dtype=np.float32)
            iou_mat  = compute_iou_matrix(gt_arr, pred_arr)
        else:
            iou_mat = np.zeros((n_gt, len(preds_sorted)), dtype=np.float32)

        precomputed.append({
            "gt_labels":   gt_labels,
            "n_gt":        n_gt,
            "pred_confs":  pred_confs,
            "pred_labels": pred_labels,
            "iou_mat":     iou_mat,
        })
    return precomputed


# ---------------------------------------------------------------------------
# Confidence sweep
# ---------------------------------------------------------------------------

def sweep_pr_curve(precomputed: list, conf_values: np.ndarray) -> tuple:
    n_steps = len(conf_values)
    macro_P = np.full(n_steps, np.nan)
    macro_R = np.full(n_steps, np.nan)

    for i, conf_thresh in enumerate(conf_values):
        matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)

        for entry in precomputed:
            gt_labels = entry["gt_labels"]
            n_gt      = entry["n_gt"]
            if n_gt == 0:
                continue

            first_active = int(np.searchsorted(entry["pred_confs"], conf_thresh))
            n_pred_all   = len(entry["pred_confs"])

            if first_active >= n_pred_all:
                for gl in gt_labels:
                    matrix[2, gl] += 1
                continue

            active_local = np.arange(first_active, n_pred_all)
            iou_sub      = entry["iou_mat"][:, active_local]
            assignments  = greedy_match(iou_sub)

            for gi, gl in enumerate(gt_labels):
                if gi in assignments:
                    pred_label = entry["pred_labels"][active_local[assignments[gi]]]
                    matrix[pred_label, gl] += 1
                else:
                    matrix[2, gl] += 1

        class_P: list = []
        class_R: list = []
        for c in range(N_CLASSES):
            tp = int(matrix[c, c])
            fp = int(matrix[c, :].sum() - tp)
            fn = int(matrix[:, c].sum() - tp)
            class_P.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
            class_R.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)

        macro_P[i] = float(np.nanmean(class_P))
        macro_R[i] = float(np.nanmean(class_R))

        if (i + 1) % 25 == 0 or (i + 1) == n_steps:
            _status(f"  Sweeping conf step {i + 1}/{n_steps}...")

    _status_end()
    return macro_P, macro_R


# ---------------------------------------------------------------------------
# Sweep cache
# ---------------------------------------------------------------------------

def _sweep_cache_path(
    model_name: str, split: str, steps: int, conf_min: float, conf_max: float
) -> Path:
    tag = f"steps{steps}_cmin{conf_min:.3f}_cmax{conf_max:.3f}".replace(".", "p")
    return WEIGHTS_DIR / f"sweep_cache_{model_name}_{split}_{tag}.npz"


def load_sweep_cache(
    model_name: str, split: str, steps: int, conf_min: float, conf_max: float
) -> tuple | None:
    path = _sweep_cache_path(model_name, split, steps, conf_min, conf_max)
    if not path.exists():
        return None
    try:
        d = np.load(path)
        print(f"  Loaded sweep cache: {path.name}")
        return d["conf_values"], d["macro_P"], d["macro_R"]
    except Exception as exc:
        print(f"  [WARN] Sweep cache unreadable ({exc}) — re-sweeping")
        path.unlink(missing_ok=True)
        return None


def save_sweep_cache(
    model_name: str,
    split: str,
    steps: int,
    conf_min: float,
    conf_max: float,
    conf_values: np.ndarray,
    macro_P: np.ndarray,
    macro_R: np.ndarray,
) -> None:
    path = _sweep_cache_path(model_name, split, steps, conf_min, conf_max)
    np.savez_compressed(path, conf_values=conf_values, macro_P=macro_P, macro_R=macro_R)
    print(f"  Saved sweep cache: {path.name}")


# ---------------------------------------------------------------------------
# Model discovery — finds .pt files in WEIGHTS_DIR
# ---------------------------------------------------------------------------

def discover_models(model_filter: str | None) -> list[tuple[str, Path]]:
    all_models = {
        p.stem: p
        for p in sorted(WEIGHTS_DIR.glob("*.pt"))
        if "_2" not in p.stem
    }
    if not all_models:
        raise FileNotFoundError(f"No .pt files found in {WEIGHTS_DIR}")
    if model_filter:
        if model_filter not in all_models:
            raise ValueError(f"No '{model_filter}.pt' found in {WEIGHTS_DIR}")
        return [(model_filter, all_models[model_filter])]
    ordered = [(n, all_models[n]) for n in MODEL_ORDER if n in all_models]
    extras  = [(n, all_models[n]) for n in sorted(all_models) if n not in MODEL_ORDER]
    return ordered + extras


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_and_save(
    model_name: str,
    split: str,
    conf_values: np.ndarray,
    macro_P: np.ndarray,
    macro_R: np.ndarray,
    best_conf: float | None,
    smooth_sigma: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    plot_R = _gaussian_smooth(macro_R, smooth_sigma)
    plot_P = _gaussian_smooth(macro_P, smooth_sigma)

    points   = np.column_stack([plot_R, plot_P]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=float(conf_values.min()), vmax=float(conf_values.max()))
    lc = LineCollection(segments, cmap=PLOT_COLORMAP, norm=norm,
                        linewidth=PLOT_LINEWIDTH, alpha=PLOT_ALPHA, zorder=3)
    lc.set_array((conf_values[:-1] + conf_values[1:]) / 2.0)
    ax.add_collection(lc)

    if best_conf is not None:
        idx = int(np.argmin(np.abs(conf_values - best_conf)))
        ax.plot(
            float(plot_R[idx]), float(plot_P[idx]),
            marker=PLOT_MARKER, markersize=PLOT_MARKER_SIZE, color=PLOT_MARKER_COLOR,
            zorder=6, linestyle="None",
            label="Best",
        )
        ax.legend(loc="lower left", fontsize=PLOT_LEGEND_FONTSIZE, framealpha=0.85)

    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_xlabel("Recall", fontsize=PLOT_XLABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.grid(True, alpha=PLOT_GRID_ALPHA, linestyle=PLOT_GRID_LINESTYLE,
            linewidth=PLOT_GRID_LINEWIDTH)
    ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_pr_curve_{split}"
    for ext in (".png", ".pdf"):
        out = out_dir / f"{stem}{ext}"
        fig.savefig(out, dpi=PLOT_DPI)
        print(f"  Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot YOLO PR curves from .pt files in the weights directory"
    )
    parser.add_argument("--model",    type=str,   default=None,
                        help="Process only this model name (e.g. yolo8n)")
    parser.add_argument("--split",    type=str,   default="test",
                        help="Split label used in output filenames (default: test)")
    parser.add_argument("--steps",    type=int,   default=2000,
                        help="Confidence sweep steps (default: 2000)")
    parser.add_argument("--conf-min", type=float, default=0.001,
                        help="Lower confidence bound (default: 0.001)")
    parser.add_argument("--conf-max", type=float, default=0.999,
                        help="Upper confidence bound (default: 0.999)")
    parser.add_argument("--smooth",   type=float, default=PLOT_SMOOTH_SIGMA,
                        help="Gaussian smoothing sigma (default: %(default)s, 0 = off)")
    parser.add_argument("--out-dir",  type=str,   default=None,
                        help="Output directory for PNG/PDF (default: weights dir)")
    args = parser.parse_args()

    out_dir     = Path(args.out_dir) if args.out_dir else WEIGHTS_DIR
    conf_values = np.linspace(args.conf_min, args.conf_max, args.steps)

    print("=" * 70)
    print("  YOLO PR Curve Plotter  (weights dir → dataset 2 test set)")
    print("=" * 70)
    print(f"  Weights dir : {WEIGHTS_DIR}")
    print(f"  Dataset     : {DATASET_DIR}")
    print(f"  Test images : {TEST_IMAGES_DIR}")
    print(f"  Split label : {args.split}")
    print(f"  Steps       : {args.steps}")
    print(f"  Conf range  : [{args.conf_min:.3f}, {args.conf_max:.3f}]")
    print(f"  Out dir     : {out_dir}")

    print("\nLocating test image-label pairs...")
    image_paths, label_paths = find_test_pairs()
    print(f"  {len(image_paths)} pairs found")

    models = discover_models(args.model)
    print(f"\n{len(models)} model(s): {[m[0] for m in models]}\n")

    ap_summary: dict = {}

    for model_name, model_path in models:
        print(f"{'─' * 70}")
        print(f"  {model_name}  ←  {model_path.name}")

        # 1. Pred cache (GPU inference, one pass per model)
        cached_data = load_or_create_pred_cache(
            model_name, model_path, image_paths, label_paths
        )

        # 2. Best confidence for star marker — prefer JSON, fall back to hard-coded table
        best_conf = MODEL_CONF_THRESH.get(model_name)
        best_json = WEIGHTS_DIR / f"best_{model_name}.json"
        if best_json.exists():
            try:
                with best_json.open("r", encoding="utf-8") as f:
                    best_conf = float(json.load(f)["best"]["conf_thresh"])
                print(f"  Best conf (from JSON): {best_conf:.4f}")
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

        # 3. Sweep cache (confidence sweep)
        sweep_cached = load_sweep_cache(
            model_name, args.split, args.steps, args.conf_min, args.conf_max
        )
        if sweep_cached is not None:
            conf_values_out, macro_P, macro_R = sweep_cached
        else:
            print("  Precomputing IoU matrices...")
            precomputed = precompute_iou_matrices(cached_data)
            print(f"  Sweeping {args.steps} confidence thresholds...")
            macro_P, macro_R = sweep_pr_curve(precomputed, conf_values)
            conf_values_out  = conf_values
            save_sweep_cache(
                model_name, args.split, args.steps, args.conf_min, args.conf_max,
                conf_values_out, macro_P, macro_R,
            )

        # 4. AP
        ap = compute_ap(macro_R, macro_P)
        ap_summary[model_name] = ap
        print(f"  AP (macro-avg): {ap:.4f}")

        # 5. Plot
        plot_and_save(
            model_name, args.split, conf_values_out, macro_P, macro_R,
            best_conf, args.smooth, out_dir,
        )

    print(f"\n{'=' * 70}")
    print(f"  AP Summary  (split={args.split})")
    print(f"{'=' * 70}")
    order_map = {n: i for i, n in enumerate(MODEL_ORDER)}
    for name, ap_val in sorted(ap_summary.items(), key=lambda x: order_map.get(x[0], 99)):
        print(f"  {name:<15}  AP = {ap_val:.4f}")


if __name__ == "__main__":
    main()
