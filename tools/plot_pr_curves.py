"""
Precision-Recall curve generator for YOLO detection models (test set).

For each discovered model checkpoint, runs a single low-confidence inference
pass on the test set (conf=0.001), saves a per-model cache, then sweeps
confidence thresholds to build a macro-averaged PR curve colored by confidence.

Runtime optimisation
--------------------
* GPU inference runs ONCE per model (conf=0.001). Results are saved to
  pred_cache_{model}_test_imgsz640.json and reused on every subsequent run.
* IoU matrices are precomputed ONCE per image (vectorised numpy). The
  confidence sweep then only re-filters the precomputed matrix — no box
  arithmetic is repeated across the 500 threshold steps.

Output per model (saved to runs/detect/weights/)
-------------------------------------------------
  {model}_pr_curve.png
  {model}_pr_curve.pdf

AP values are printed to the console after each model.

Usage
-----
    python tools/plot_pr_curves.py                   # all models
    python tools/plot_pr_curves.py --model yolo8n    # one model
    python tools/plot_pr_curves.py --steps 300       # fewer steps
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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGES_DIR = PROJECT_ROOT / "dataset" / "test 2" / "images"
TEST_LABELS_DIR = PROJECT_ROOT / "dataset" / "test 2" / "labels"
TEST_TAG = TEST_IMAGES_DIR.parent.name.replace(" ", "")  # e.g. "test2"
DETECT_DIR = PROJECT_ROOT / "runs" / "detect"
WEIGHTS_DIR = DETECT_DIR / "weights"
FASTERRCNN_TRAIN_DIR = PROJECT_ROOT / "runs" / "fasterrcnn" / "train"

CLASS_NAMES = ["bird", "drone", "unknown"]
N_CLASSES = len(CLASS_NAMES)

IOU_THRESH = 0.5
IMGSZ = 640

# ---------------------------------------------------------------------------
# Plot style — edit these to change the appearance without touching the code
# ---------------------------------------------------------------------------

PLOT_FIGSIZE          = (7, 6)      # (width, height) in inches
PLOT_LINEWIDTH        = 2.5         # thickness of the PR curve line
PLOT_ALPHA            = 1        # opacity of the PR curve line
PLOT_COLORMAP         = "plasma"    # matplotlib colormap for confidence coloring
PLOT_SMOOTH_SIGMA     = 9.0         # Gaussian smoothing sigma (0 = off)

PLOT_XLIM             = (0.0, 1.0)  # (min, max) for the Recall axis
PLOT_YLIM             = (0.0, 1.05) # (min, max) for the Precision axis

PLOT_XLABEL_FONTSIZE  = 20
PLOT_YLABEL_FONTSIZE  = 20
PLOT_TITLE_FONTSIZE   = 13
PLOT_TICK_FONTSIZE    = 10
PLOT_LEGEND_FONTSIZE  = 20

PLOT_GRID_ALPHA       = 0.3         # opacity of the background grid
PLOT_GRID_LINESTYLE   = "--"
PLOT_GRID_LINEWIDTH   = 0.7

PLOT_MARKER           = "$*$"       # best-point marker ('$*$' = asterisk, '*' = star shape)
PLOT_MARKER_SIZE      = 20
PLOT_MARKER_COLOR     = "black"

PLOT_DPI              = 150         # output resolution for PNG
CONF_INFER = 0.001

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n"]

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
# Core geometry helpers
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
    gt_e = gt[:, np.newaxis, :]     # (n, 1, 4)
    pred_e = pred[np.newaxis, :, :]  # (1, m, 4)

    x1 = np.maximum(gt_e[..., 0], pred_e[..., 0])
    y1 = np.maximum(gt_e[..., 1], pred_e[..., 1])
    x2 = np.minimum(gt_e[..., 2], pred_e[..., 2])
    y2 = np.minimum(gt_e[..., 3], pred_e[..., 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_gt = (
        np.maximum(0.0, gt_e[..., 2] - gt_e[..., 0])
        * np.maximum(0.0, gt_e[..., 3] - gt_e[..., 1])
    )
    area_pred = (
        np.maximum(0.0, pred_e[..., 2] - pred_e[..., 0])
        * np.maximum(0.0, pred_e[..., 3] - pred_e[..., 1])
    )
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
    boxes: list = []
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

def _build_test_cache(
    model_path: Path,
    image_paths: list,
    label_paths: list,
    cache_path: Path,
) -> list:
    """Run model at CONF_INFER on every test image and save predictions."""
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

        result = model.predict(
            source=str(ip), conf=CONF_INFER, imgsz=IMGSZ, verbose=False
        )
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
        m, s = divmod(rem, 60)
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
    print(f"  Saved test cache: {cache_path.name}  ({total} images)")
    return cached


def load_or_create_cache(
    model_name: str,
    model_path: Path,
    image_paths: list,
    label_paths: list,
) -> list:
    cache_path = WEIGHTS_DIR / f"pred_cache_{model_name}_{TEST_TAG}_imgsz{IMGSZ}.json"
    if cache_path.exists():
        try:
            print(f"  Loading test cache: {cache_path.name}")
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Cache ready  ({len(data)} images)")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Cache corrupt ({exc}) — re-running inference")
            cache_path.unlink(missing_ok=True)
    return _build_test_cache(model_path, image_paths, label_paths, cache_path)


# ---------------------------------------------------------------------------
# IoU precomputation (one-time per model, before the conf sweep)
# ---------------------------------------------------------------------------

def precompute_iou_matrices(cached_data: list) -> list:
    """
    For each image, build the full IoU matrix (GT x all_preds at conf=0.001)
    once. The confidence sweep slices this matrix rather than recomputing IoU,
    giving a large speedup over the naive per-threshold approach.
    """
    precomputed: list = []
    for entry in cached_data:
        gt_labels = entry["gt_labels"]
        n_gt = len(gt_labels)

        # Sort predictions by confidence ascending so searchsorted works correctly
        preds_sorted = sorted(entry["preds"], key=lambda x: x[0])
        pred_confs = np.array([p[0] for p in preds_sorted], dtype=np.float32)
        pred_labels = [p[1] for p in preds_sorted]

        if n_gt > 0 and len(preds_sorted) > 0:
            gt_arr = np.array(entry["gt_boxes"], dtype=np.float32)
            pred_arr = np.array([p[2] for p in preds_sorted], dtype=np.float32)
            iou_mat = compute_iou_matrix(gt_arr, pred_arr)  # (n_gt, n_pred)
        else:
            iou_mat = np.zeros((n_gt, len(preds_sorted)), dtype=np.float32)

        precomputed.append(
            {
                "gt_labels": gt_labels,
                "n_gt": n_gt,
                "pred_confs": pred_confs,
                "pred_labels": pred_labels,
                "iou_mat": iou_mat,
            }
        )
    return precomputed


# ---------------------------------------------------------------------------
# Confidence sweep
# ---------------------------------------------------------------------------

def sweep_pr_curve(precomputed: list, conf_values: np.ndarray) -> tuple:
    """
    Sweep confidence thresholds and compute macro-averaged P/R at each step.
    IoU matrices are pre-cached; only filtering (searchsorted) and greedy
    matching happen here — no box arithmetic at all inside the loop.
    """
    n_steps = len(conf_values)
    macro_P = np.full(n_steps, np.nan)
    macro_R = np.full(n_steps, np.nan)

    for i, conf_thresh in enumerate(conf_values):
        matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)

        for entry in precomputed:
            gt_labels = entry["gt_labels"]
            n_gt = entry["n_gt"]
            if n_gt == 0:
                continue

            # pred_confs is sorted ascending; searchsorted gives first index >= conf_thresh
            first_active = int(np.searchsorted(entry["pred_confs"], conf_thresh))
            n_pred_all = len(entry["pred_confs"])

            if first_active >= n_pred_all:
                # No predictions pass the threshold — all GT boxes are unmatched
                for gl in gt_labels:
                    matrix[2, gl] += 1
                continue

            active_local = np.arange(first_active, n_pred_all)
            iou_sub = entry["iou_mat"][:, active_local]  # (n_gt, n_active)
            assignments = greedy_match(iou_sub)

            for gi, gl in enumerate(gt_labels):
                if gi in assignments:
                    pred_label = entry["pred_labels"][active_local[assignments[gi]]]
                    matrix[pred_label, gl] += 1
                else:
                    matrix[2, gl] += 1

        # Macro-averaged precision and recall over all N_CLASSES
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
# Sweep cache  (saves macro_P / macro_R so re-plotting is instant)
# ---------------------------------------------------------------------------

def _sweep_cache_path(model_name: str, steps: int, conf_min: float, conf_max: float) -> Path:
    tag = f"steps{steps}_cmin{conf_min:.3f}_cmax{conf_max:.3f}".replace(".", "p")
    return WEIGHTS_DIR / f"sweep_cache_{model_name}_{TEST_TAG}_{tag}.npz"


def load_sweep_cache(
    model_name: str, steps: int, conf_min: float, conf_max: float
) -> tuple | None:
    path = _sweep_cache_path(model_name, steps, conf_min, conf_max)
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
# Metrics
# ---------------------------------------------------------------------------

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the PR curve via trapezoidal integration (recall ascending)."""
    valid = ~(np.isnan(recall) | np.isnan(precision))
    r = recall[valid]
    p = precision[valid]
    if len(r) < 2:
        return float("nan")
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1-D array with a Gaussian kernel (pure numpy, no scipy).
    Uses reflection padding so the curve does not drift at the edges."""
    if sigma <= 0:
        return arr
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Reflect-pad both ends so edge values don't pull toward zero
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
    fig, ax = plt.subplots(figsize=(7, 6))

    # Smooth for display only — raw values are used for AP and the star marker
    plot_R = _gaussian_smooth(macro_R, smooth_sigma)
    plot_P = _gaussian_smooth(macro_P, smooth_sigma)

    # --- Confidence-colored PR curve via LineCollection ---
    # High conf -> top-left (high P, low R); Low conf -> bottom-right (low P, high R)
    # plasma: dark purple = low conf, bright yellow = high conf
    points = np.column_stack([plot_R, plot_P]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=float(conf_values.min()), vmax=float(conf_values.max()))
    cmap = plt.cm.plasma

    # Color each segment by its midpoint confidence
    seg_confs = (conf_values[:-1] + conf_values[1:]) / 2.0
    lc = LineCollection(
        segments, cmap=cmap, norm=norm, linewidth=2.5, alpha=0.95, zorder=3
    )
    lc.set_array(seg_confs)
    ax.add_collection(lc)

    # --- Colorbar --- (removed)

    # --- Best operating point star (plotted on smoothed curve) ---
    if best_conf is not None:
        idx = int(np.argmin(np.abs(conf_values - best_conf)))
        bx, by = float(plot_R[idx]), float(plot_P[idx])
        ax.plot(
            bx, by,
            marker="$*$",
            markersize=20,
            color="black",
            zorder=6,
            linestyle="None",
            label=f"Best  (conf = {best_conf:.3f})",
        )
        ax.legend(loc="lower left", fontsize=10, framealpha=0.85)

    # --- Axes ---
    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(
        f"{display_name}  —  Precision–Recall Curve\n"
        f"Macro-avg AP = {ap:.4f}  |  Test set",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    ax.tick_params(labelsize=10)

    fig.tight_layout()

    for ext in (".png", ".pdf"):
        out = WEIGHTS_DIR / f"{model_name}_pr_curve{ext}"
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out.name}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models(requested: str | None) -> list:
    """Find YOLO models from runs/detect/weights/*.pt (excluding '_2' variants and fasterrcnn)."""
    all_models = {
        p.stem: p
        for p in sorted(WEIGHTS_DIR.glob("*.pt"))
        if "_2" not in p.stem and p.stem != "fasterrcnn"
    }
    if not all_models:
        raise FileNotFoundError(f"No .pt files found in {WEIGHTS_DIR}")
    if requested:
        if requested not in all_models:
            raise ValueError(f"No '{requested}.pt' found in {WEIGHTS_DIR}")
        return [(requested, all_models[requested])]
    ordered = [(n, all_models[n]) for n in MODEL_ORDER if n in all_models]
    extras = [(n, all_models[n]) for n in sorted(all_models) if n not in MODEL_ORDER]
    return ordered + extras


# ---------------------------------------------------------------------------
# Faster R-CNN helpers
# ---------------------------------------------------------------------------

def _load_fasterrcnn(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cls_weight = checkpoint["model_state_dict"]["roi_heads.box_predictor.cls_score.weight"]
    num_classes = cls_weight.shape[0]
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _infer_fasterrcnn_image(model, image_path: Path, device: str) -> list:
    image = Image.open(image_path).convert("RGB")
    tensor = (
        torch.from_numpy(np.array(image, dtype="uint8"))
        .permute(2, 0, 1).float().div(255.0).to(device)
    )
    with torch.no_grad():
        pred = model([tensor])[0]
    results = []
    for box, label, score in zip(
        pred["boxes"].cpu().numpy(),
        pred["labels"].cpu().numpy(),
        pred["scores"].cpu().numpy(),
    ):
        shifted = int(label) - 1  # torchvision: 0=background -> shift to 0=bird
        cls = shifted if shifted in (0, 1) else 2
        results.append((float(score), cls, [float(v) for v in box]))
    return results


def _build_test_cache_fasterrcnn(
    checkpoint_path: Path,
    image_paths: list,
    label_paths: list,
    cache_path: Path,
    device: str,
) -> list:
    print(f"  Loading FasterRCNN weights: {checkpoint_path.name}")
    model = _load_fasterrcnn(checkpoint_path, device)
    total = len(image_paths)
    cached: list = []
    t0 = time.time()
    for idx, (ip, lp) in enumerate(zip(image_paths, label_paths), 1):
        gt_boxes_xywh, gt_labels = load_label_file(lp)
        with Image.open(ip) as img:
            width, height = img.size
        gt_boxes_xyxy = [xywhn_to_xyxy(b, width, height) for b in gt_boxes_xywh]
        preds = _infer_fasterrcnn_image(model, ip, device)
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
    print(f"  Saved FasterRCNN test cache: {cache_path.name}  ({total} images)")
    return cached


def load_or_create_cache_fasterrcnn(
    checkpoint_path: Path,
    image_paths: list,
    label_paths: list,
    device: str,
) -> list:
    cache_path = WEIGHTS_DIR / f"pred_cache_fasterrcnn_{TEST_TAG}_imgsz640.json"
    if cache_path.exists():
        try:
            print(f"  Loading FasterRCNN test cache: {cache_path.name}")
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Cache ready  ({len(data)} images)")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Cache corrupt ({exc}) — re-running inference")
            cache_path.unlink(missing_ok=True)
    return _build_test_cache_fasterrcnn(checkpoint_path, image_paths, label_paths, cache_path, device)


def _fasterrcnn_best_conf() -> float | None:
    best_json = FASTERRCNN_TRAIN_DIR.parent / "tune_f1" / "best_fasterrcnn_f1.json"
    if not best_json.exists():
        return None
    try:
        with best_json.open(encoding="utf-8") as f:
            entries = json.load(f)
        if isinstance(entries, list) and entries:
            best = max(entries, key=lambda x: x.get("f1", 0.0))
            return float(best["conf_thresh"])
    except (json.JSONDecodeError, OSError, KeyError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PR curves for YOLO models on the test set"
    )
    parser.add_argument("--model", type=str, default=None, help="Only process one model name")
    parser.add_argument("--steps", type=int, default=2000, help="Confidence sweep steps")
    parser.add_argument("--conf-min", type=float, default=0.001, help="Lower conf bound")
    parser.add_argument("--conf-max", type=float, default=0.999, help="Upper conf bound")
    parser.add_argument("--smooth", type=float, default=8.0,
                        help="Gaussian smoothing sigma for the plot (0 = off)")
    args = parser.parse_args()

    conf_values = np.linspace(args.conf_min, args.conf_max, args.steps)

    print("=" * 70)
    print("  YOLO Precision-Recall Curve Generator  (test set)")
    print("=" * 70)
    print(f"  Test images : {TEST_IMAGES_DIR}")
    print(f"  Weights dir : {WEIGHTS_DIR}")
    print(f"  Conf sweep  : [{args.conf_min:.3f}, {args.conf_max:.3f}]  x  {args.steps} steps")
    print(f"  IOU thresh  : {IOU_THRESH}")

    print("\nLocating test image-label pairs...")
    image_paths, label_paths = find_test_pairs()
    print(f"  {len(image_paths)} pairs found")

    models = discover_models(args.model)
    print(f"\n{len(models)} model(s): {[m[0] for m in models]}\n")

    ap_summary: dict = {}

    for model_name, model_path in models:
        print(f"{'=' * 70}")
        print(f"  Model: {model_name}  ({model_path.name})")
        print(f"{'=' * 70}")

        # 1. Load or build test-set prediction cache (GPU only on first run)
        cached_data = load_or_create_cache(
            model_name, model_path, image_paths, label_paths
        )

        # 2. Load best conf from JSON for the star marker
        best_conf: float | None = None
        best_json = WEIGHTS_DIR / f"best_{model_name}.json"
        if best_json.exists():
            try:
                with best_json.open("r", encoding="utf-8") as f:
                    best_conf = float(json.load(f)["best"]["conf_thresh"])
                print(f"  Best conf (from JSON): {best_conf:.4f}")
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
        if best_conf is None:
            print("  [INFO] No best_*.json — star marker will be skipped")

        # 3. Precompute per-image IoU matrices and sweep (skipped if sweep cache exists)
        sweep_cached = load_sweep_cache(model_name, args.steps, args.conf_min, args.conf_max)
        if sweep_cached is not None:
            conf_values, macro_P, macro_R = sweep_cached
        else:
            print("  Precomputing IoU matrices...")
            precomputed = precompute_iou_matrices(cached_data)

            # 4. Sweep confidence thresholds (CPU-only, fast)
            print(f"  Sweeping {args.steps} confidence thresholds...")
            macro_P, macro_R = sweep_pr_curve(precomputed, conf_values)
            save_sweep_cache(model_name, args.steps, args.conf_min, args.conf_max,
                             conf_values, macro_P, macro_R)

        # 5. Compute and report AP
        ap = compute_ap(macro_R, macro_P)
        ap_summary[model_name] = ap
        print(f"  AP (macro-avg, test set): {ap:.4f}")

        # 6. Plot and save PNG + PDF
        print("  Generating plots...")
        plot_and_save(model_name, conf_values, macro_P, macro_R, ap, best_conf,
                      smooth_sigma=args.smooth)
        print()

    # --- Faster R-CNN ---
    if args.model is None or args.model == "fasterrcnn":
        frcnn_pt = FASTERRCNN_TRAIN_DIR / "best.pt"
        if frcnn_pt.exists():
            print(f"\n{'=' * 70}")
            print(f"  Model: fasterrcnn  ({frcnn_pt.name})")
            print(f"{'=' * 70}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            frcnn_cached = load_or_create_cache_fasterrcnn(frcnn_pt, image_paths, label_paths, device)
            frcnn_best_conf = _fasterrcnn_best_conf()
            if frcnn_best_conf is not None:
                print(f"  Best conf (from JSON): {frcnn_best_conf:.4f}")
            else:
                print("  [INFO] No best conf JSON — star marker will be skipped")
            frcnn_sweep = load_sweep_cache("fasterrcnn", args.steps, args.conf_min, args.conf_max)
            if frcnn_sweep is not None:
                frcnn_confs, frcnn_P, frcnn_R = frcnn_sweep
            else:
                print("  Precomputing IoU matrices...")
                frcnn_precomputed = precompute_iou_matrices(frcnn_cached)
                print(f"  Sweeping {args.steps} confidence thresholds...")
                frcnn_P, frcnn_R = sweep_pr_curve(frcnn_precomputed, conf_values)
                frcnn_confs = conf_values
                save_sweep_cache("fasterrcnn", args.steps, args.conf_min, args.conf_max,
                                 frcnn_confs, frcnn_P, frcnn_R)
            frcnn_ap = compute_ap(frcnn_R, frcnn_P)
            ap_summary["fasterrcnn"] = frcnn_ap
            print(f"  AP (macro-avg, test set): {frcnn_ap:.4f}")
            print("  Generating plots...")
            plot_and_save("fasterrcnn", frcnn_confs, frcnn_P, frcnn_R, frcnn_ap,
                          frcnn_best_conf, smooth_sigma=args.smooth)
            print()
        else:
            print(f"\n[SKIP] fasterrcnn: {frcnn_pt} not found")

    # Final AP ranking
    print(f"{'=' * 70}")
    print("  AP Summary  —  macro-avg, test set")
    print(f"{'=' * 70}")
    for name, ap_val in sorted(ap_summary.items(), key=lambda x: -x[1]):
        print(f"  {name:<15}  AP = {ap_val:.4f}")


if __name__ == "__main__":
    main()
