"""
Precision-Recall curve generator for YOLO models.

For each enabled model, runs inference at low confidence (conf=0.001), saves a
per-model prediction cache, then sweeps confidence thresholds to build a
macro-averaged PR curve colored by confidence.

Caches stored in   runs/detect/{model}/eval_cache/
PR curves (PNG+PDF) saved to  runs/detect/{model}/

Usage
-----
    python tools/plot_yolo_pr_curves.py                  # all enabled models
    python tools/plot_yolo_pr_curves.py --model yolo8n   # one model
    python tools/plot_yolo_pr_curves.py --steps 300      # fewer steps
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
# Paths & dataset
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECT_DIR   = PROJECT_ROOT / "runs" / "detect"

# Toggle between test and validation dataset
USE_TEST_DATASET = True    # True = test, False = validation
DATASET_SPLIT    = "test" if USE_TEST_DATASET else "validation"
IMAGES_DIR       = PROJECT_ROOT / "dataset" / DATASET_SPLIT / "images"
LABELS_DIR       = PROJECT_ROOT / "dataset" / DATASET_SPLIT / "labels"

# ---------------------------------------------------------------------------
# Classes & eval settings
# ---------------------------------------------------------------------------

CLASS_NAMES = ["bird", "drone", "unknown"]
N_CLASSES   = len(CLASS_NAMES)

IOU_THRESH = 0.5
IMGSZ      = 640
CONF_INFER = 0.001   # low-conf inference pass to capture all detections

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

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n"]

# Set RUN_ALL_MODELS = True to run every model regardless of ENABLED_MODELS.
RUN_ALL_MODELS = True
ENABLED_MODELS = {
    "yolo8n":  False,
    "yolo8m":  False,
    "yolo9t":  False,
    "yolo10n": False,
    "yolo11n": False,
    "yolo12n": False,
    "yolo26n": False,
}

# ---------------------------------------------------------------------------
# Plot style — edit these to change the appearance without touching the code
# ---------------------------------------------------------------------------

PLOT_FIGSIZE          = (7, 6)      # (width, height) in inches
PLOT_LINEWIDTH        = 2.5         # thickness of the PR curve line
PLOT_ALPHA            = 1.0         # opacity of the PR curve line
PLOT_COLORMAP         = "plasma"    # matplotlib colormap for confidence coloring
PLOT_SMOOTH_SIGMA     = 9.0         # Gaussian smoothing sigma (0 = off)

PLOT_XLIM             = (0.0, 1.0)  # (min, max) for the Recall axis
PLOT_YLIM             = (0.0, 1.05) # (min, max) for the Precision axis

PLOT_XLABEL_FONTSIZE  = 20
PLOT_YLABEL_FONTSIZE  = 20
PLOT_TITLE_FONTSIZE   = 20
PLOT_TICK_FONTSIZE    = 20
PLOT_LEGEND_FONTSIZE  = 20

PLOT_GRID_ALPHA       = 0.3         # opacity of the background grid
PLOT_GRID_LINESTYLE   = "--"
PLOT_GRID_LINEWIDTH   = 0.7

PLOT_MARKER           = "$*$"       # best-point marker ('$*$' = asterisk, '*' = star shape)
PLOT_MARKER_SIZE      = 20
PLOT_MARKER_COLOR     = "black"

PLOT_DPI              = 150         # output resolution for PNG

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

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
# Geometry helpers
# ---------------------------------------------------------------------------

def xywhn_to_xyxy(box: tuple, w: int, h: int) -> list:
    cx, cy, bw, bh = box
    return [
        (cx - bw / 2.0) * w, (cy - bh / 2.0) * h,
        (cx + bw / 2.0) * w, (cy + bh / 2.0) * h,
    ]


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_e   = gt[:, np.newaxis, :]
    pred_e = pred[np.newaxis, :, :]
    x1 = np.maximum(gt_e[..., 0], pred_e[..., 0])
    y1 = np.maximum(gt_e[..., 1], pred_e[..., 1])
    x2 = np.minimum(gt_e[..., 2], pred_e[..., 2])
    y2 = np.minimum(gt_e[..., 3], pred_e[..., 3])
    inter     = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_gt   = np.maximum(0.0, gt_e[..., 2]  - gt_e[..., 0])  * np.maximum(0.0, gt_e[..., 3]  - gt_e[..., 1])
    area_pred = np.maximum(0.0, pred_e[..., 2] - pred_e[..., 0]) * np.maximum(0.0, pred_e[..., 3] - pred_e[..., 1])
    union = area_gt + area_pred - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(iou_sub: np.ndarray) -> dict:
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


def find_image_label_pairs() -> tuple[list, list]:
    image_paths: list = []
    label_paths: list = []
    for lp in sorted(LABELS_DIR.glob("*.txt")):
        for ext in IMAGE_EXTENSIONS:
            ip = IMAGES_DIR / f"{lp.stem}{ext}"
            if ip.exists():
                image_paths.append(ip)
                label_paths.append(lp)
                break
    if not image_paths:
        raise FileNotFoundError(
            f"No image-label pairs found.\n  Images: {IMAGES_DIR}\n  Labels: {LABELS_DIR}"
        )
    return image_paths, label_paths


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models(requested: str | None) -> list[tuple[str, Path]]:
    """Return list of (model_name, model_pt) from runs/detect/{model}/weights/best.pt."""
    if requested is not None:
        pt = DETECT_DIR / requested / "weights" / "best.pt"
        if not pt.exists():
            raise FileNotFoundError(f"No best.pt found for '{requested}' at {pt}")
        return [(requested, pt)]

    if RUN_ALL_MODELS:
        names = MODEL_ORDER
    else:
        names = [n for n in MODEL_ORDER if ENABLED_MODELS.get(n, False)]
        if not names:
            raise ValueError(
                "No models enabled. Set RUN_ALL_MODELS=True or enable at least one in ENABLED_MODELS."
            )

    result = []
    for name in names:
        pt = DETECT_DIR / name / "weights" / "best.pt"
        if pt.exists():
            result.append((name, pt))
        else:
            print(f"  [SKIP] {name}: no best.pt at {pt}")
    if not result:
        raise FileNotFoundError(f"No best.pt found under {DETECT_DIR}/<model>/weights/")
    return result


# ---------------------------------------------------------------------------
# Prediction caching
# ---------------------------------------------------------------------------

def _pred_cache_path(model_name: str) -> Path:
    cache_dir = DETECT_DIR / model_name / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"pred_cache_{model_name}_{DATASET_SPLIT}_imgsz{IMGSZ}.json"


def _build_pred_cache(
    model_path: Path,
    image_paths: list,
    label_paths: list,
    cache_path: Path,
) -> list:
    print(f"  Loading model weights: {model_path.name}")
    model  = YOLO(str(model_path))
    total  = len(image_paths)
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
                preds.append((float(conf), cls_id if cls_id in (0, 1) else 2, [float(v) for v in box]))

        cached.append({"gt_boxes": gt_boxes_xyxy, "gt_labels": gt_labels, "preds": preds})

        elapsed = time.time() - t0
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        _status(f"  Caching {idx}/{total} ({idx / total * 100:.1f}%)  |  elapsed {h}:{m:02d}:{s:02d}")

    _status_end()
    del model
    torch.cuda.empty_cache()

    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cached, f)
    tmp.replace(cache_path)
    print(f"  Saved pred cache: {cache_path.name}  ({total} images)")
    return cached


def load_or_build_pred_cache(
    model_name: str,
    model_path: Path,
    image_paths: list,
    label_paths: list,
) -> list:
    path = _pred_cache_path(model_name)
    if path.exists():
        try:
            print(f"  Loading pred cache: {path.name}")
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Cache ready  ({len(data)} images)")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Cache corrupt ({exc}) -- re-running inference")
            path.unlink(missing_ok=True)
    return _build_pred_cache(model_path, image_paths, label_paths, path)


# ---------------------------------------------------------------------------
# IoU precomputation
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
            gt_arr   = np.array(entry["gt_boxes"],            dtype=np.float32)
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

def _sweep_cache_path(model_name: str, steps: int, conf_min: float, conf_max: float) -> Path:
    tag       = f"steps{steps}_cmin{conf_min:.3f}_cmax{conf_max:.3f}".replace(".", "p")
    cache_dir = DETECT_DIR / model_name / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"sweep_cache_{model_name}_{DATASET_SPLIT}_{tag}.npz"


def load_sweep_cache(model_name: str, steps: int, conf_min: float, conf_max: float) -> tuple | None:
    path = _sweep_cache_path(model_name, steps, conf_min, conf_max)
    if not path.exists():
        return None
    try:
        d = np.load(path)
        print(f"  Loaded sweep cache: {path.name}")
        return d["conf_values"], d["macro_P"], d["macro_R"]
    except Exception as exc:
        print(f"  [WARN] Sweep cache unreadable ({exc}) -- re-sweeping")
        path.unlink(missing_ok=True)
        return None


def save_sweep_cache(
    model_name: str,
    steps: int,
    conf_min: float,
    conf_max: float,
    conf_values: np.ndarray,
    macro_P: np.ndarray,
    macro_R: np.ndarray,
) -> None:
    path = _sweep_cache_path(model_name, steps, conf_min, conf_max)
    np.savez_compressed(path, conf_values=conf_values, macro_P=macro_P, macro_R=macro_R)
    print(f"  Saved sweep cache: {path.name}")


# ---------------------------------------------------------------------------
# AP metric
# ---------------------------------------------------------------------------

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    valid = ~(np.isnan(recall) | np.isnan(precision))
    r, p  = recall[valid], precision[valid]
    if len(r) < 2:
        return float("nan")
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


# ---------------------------------------------------------------------------
# Plotting
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


def format_display_name(name: str) -> str:
    if name.lower().startswith("yolo") and len(name) > 4 and name[4].isdigit():
        return f"YOLOv{name[4:]}"
    return name


def plot_and_save(
    model_name: str,
    conf_values: np.ndarray,
    macro_P: np.ndarray,
    macro_R: np.ndarray,
    ap: float,
    best_conf: float | None,
    smooth_sigma: float = 0.0,
) -> None:
    display_name = format_display_name(model_name)
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
            label=f"Confidence threshold: {best_conf * 100:.2f}",
        )
        ax.legend(loc="lower left", fontsize=PLOT_LEGEND_FONTSIZE, framealpha=0.85)

    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_xlabel("Recall", fontsize=PLOT_XLABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.grid(True, alpha=PLOT_GRID_ALPHA, linestyle=PLOT_GRID_LINESTYLE, linewidth=PLOT_GRID_LINEWIDTH)
    ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
    fig.tight_layout()

    out_dir = DETECT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        out = out_dir / f"{model_name}_pr_curve_aod4{ext}"
        fig.savefig(out, dpi=PLOT_DPI)
        print(f"  Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PR curves for YOLO models")
    parser.add_argument("--model",    type=str,   default=None,  help="Process only this model name")
    parser.add_argument("--steps",    type=int,   default=2000,  help="Confidence sweep steps")
    parser.add_argument("--conf-min", type=float, default=0.001, help="Lower conf bound")
    parser.add_argument("--conf-max", type=float, default=0.999, help="Upper conf bound")
    parser.add_argument("--smooth",   type=float, default=PLOT_SMOOTH_SIGMA,
                        help="Gaussian smoothing sigma (0 = off)")
    args = parser.parse_args()

    conf_values = np.linspace(args.conf_min, args.conf_max, args.steps)

    print("=" * 70)
    print("  YOLO Precision-Recall Curve Generator")
    print("=" * 70)
    print(f"  Dataset     : {DATASET_SPLIT}  ({IMAGES_DIR})")
    print(f"  Conf sweep  : [{args.conf_min:.3f}, {args.conf_max:.3f}]  x  {args.steps} steps")
    print(f"  IOU thresh  : {IOU_THRESH}")

    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABELS_DIR}")

    print("\nLocating image-label pairs...")
    image_paths, label_paths = find_image_label_pairs()
    print(f"  {len(image_paths)} pairs found")

    models = discover_models(args.model)
    print(f"\n{len(models)} model(s): {[m[0] for m in models]}\n")

    ap_summary: dict = {}

    for model_name, model_path in models:
        print(f"{'=' * 70}")
        print(f"  Model: {model_name}  ({model_path})")
        print(f"{'=' * 70}")

        cached_data = load_or_build_pred_cache(model_name, model_path, image_paths, label_paths)

        best_conf: float | None = MODEL_CONF_THRESH.get(model_name)
        if best_conf is not None:
            print(f"  Best conf (MODEL_CONF_THRESH): {best_conf:.4f}")
        else:
            print("  [INFO] No best conf -- star marker will be skipped")

        sweep_cached = load_sweep_cache(model_name, args.steps, args.conf_min, args.conf_max)
        if sweep_cached is not None:
            conf_values_out, macro_P, macro_R = sweep_cached
        else:
            print("  Precomputing IoU matrices...")
            precomputed = precompute_iou_matrices(cached_data)
            print(f"  Sweeping {args.steps} confidence thresholds...")
            macro_P, macro_R = sweep_pr_curve(precomputed, conf_values)
            conf_values_out = conf_values
            save_sweep_cache(model_name, args.steps, args.conf_min, args.conf_max,
                             conf_values_out, macro_P, macro_R)

        ap = compute_ap(macro_R, macro_P)
        ap_summary[model_name] = ap
        print(f"  AP (macro-avg, {DATASET_SPLIT}): {ap:.4f}")

        print("  Generating plots...")
        plot_and_save(model_name, conf_values_out, macro_P, macro_R, ap, best_conf,
                      smooth_sigma=args.smooth)
        print()

    print(f"{'=' * 70}")
    print(f"  AP Summary  --  macro-avg, {DATASET_SPLIT}")
    print(f"{'=' * 70}")
    for name, ap_val in sorted(ap_summary.items(), key=lambda x: -x[1]):
        print(f"  {name:<15}  AP = {ap_val:.4f}")


if __name__ == "__main__":
    main()
