"""
Precision-Recall curve generator for YOLO models using WC-NMS post-processing.

For each enabled model, runs WC-NMS inference at low confidence (conf=0.001),
saves a per-model prediction cache, then sweeps confidence thresholds to build
a macro-averaged PR curve colored by confidence.

Caches stored in   runs/detect/{model}/eval_cache/
PR curves (PNG+PDF) saved to  runs/detect/{model}/

Usage
-----
    python tools/plot_wc-nms-pr_curves.py                  # all enabled models
    python tools/plot_wc-nms-pr_curves.py --model yolo8n   # one model
    python tools/plot_wc-nms-pr_curves.py --steps 300      # fewer steps
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
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

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
CONF_INFER = 0.001   # low-conf inference pass to capture all WC-NMS detections
NMS_THRESH = 0.50    # EIoU threshold for WC-NMS clustering
MAX_DET    = 300

# ---------------------------------------------------------------------------
# Per-model star-marker confidence threshold
# WC-NMS merged scores are raw class scores (lower than post-NMS YOLO scores).
# 0.30 matches WCNMS_CONF_THRESH used in eval_wc-nms.py.
# ---------------------------------------------------------------------------

MODEL_CONF_THRESH = {
    "yolo8n":  0.30,
    "yolo8m":  0.30,
    "yolo9t":  0.30,
    "yolo10n": 0.30,
    "yolo11n": 0.30,
    "yolo12n": 0.30,
    "yolo26n": 0.30,
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
# WC-NMS inference helpers  (from eval_wc-nms.py)
# ---------------------------------------------------------------------------

def _preprocess_image(image_path: Path, device) -> tuple:
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    orig_shape = img_np.shape[:2]
    letterbox = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, scale_fill=False, scaleup=True, stride=32)
    img_lb = letterbox(image=img_np)
    img_lb = img_lb.transpose((2, 0, 1))
    img_lb = np.ascontiguousarray(img_lb)
    im = torch.from_numpy(img_lb).to(device)
    im = im.float() / 255.0
    im = im.unsqueeze(0)
    return im, orig_shape


def _unwrap_raw_predictions(raw_output):
    if isinstance(raw_output, torch.Tensor):
        return raw_output
    if isinstance(raw_output, (list, tuple)):
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
        return pred
    raise RuntimeError(f"Unsupported raw model output type: {type(raw_output)}")


def _decode_raw_predictions(raw_pred, num_classes: int):
    if raw_pred.ndim != 3 or raw_pred.shape[0] != 1:
        raise RuntimeError(f"Unexpected raw prediction shape: {tuple(raw_pred.shape)}")
    if raw_pred.shape[1] == 4 + num_classes:
        return raw_pred[0].transpose(0, 1)
    if raw_pred.shape[2] == 4 + num_classes:
        return raw_pred[0]
    raise RuntimeError(
        f"Could not interpret raw prediction shape {tuple(raw_pred.shape)} for nc={num_classes}"
    )


def _eiou_matrix(boxes):
    x1 = torch.max(boxes[:, None, 0], boxes[None, :, 0])
    y1 = torch.max(boxes[:, None, 1], boxes[None, :, 1])
    x2 = torch.min(boxes[:, None, 2], boxes[None, :, 2])
    y2 = torch.min(boxes[:, None, 3], boxes[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
            (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    union = area[:, None] + area[None, :] - inter
    iou = inter / union.clamp(min=1e-9)

    widths  = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-9)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-9)
    ctr_x   = (boxes[:, 0] + boxes[:, 2]) / 2.0
    ctr_y   = (boxes[:, 1] + boxes[:, 3]) / 2.0

    dx = ctr_x[:, None] - ctr_x[None, :]
    dy = ctr_y[:, None] - ctr_y[None, :]
    dw = (widths[:, None]  - widths[None, :]).pow(2)
    dh = (heights[:, None] - heights[None, :]).pow(2)

    enc_x1 = torch.min(boxes[:, None, 0], boxes[None, :, 0])
    enc_y1 = torch.min(boxes[:, None, 1], boxes[None, :, 1])
    enc_x2 = torch.max(boxes[:, None, 2], boxes[None, :, 2])
    enc_y2 = torch.max(boxes[:, None, 3], boxes[None, :, 3])
    wc = (enc_x2 - enc_x1).clamp(min=1e-9)
    hc = (enc_y2 - enc_y1).clamp(min=1e-9)

    r = (dx.pow(2) + dy.pow(2)) / (wc.pow(2) + hc.pow(2)).clamp(min=1e-9)
    r = r + dw / wc.pow(2).clamp(min=1e-9)
    r = r + dh / hc.pow(2).clamp(min=1e-9)

    x = iou - r
    x.fill_diagonal_(0.0)
    return x


def _wc_nms_single_class(boxes, scores, thresh):
    if boxes.numel() == 0:
        return boxes, scores, torch.empty((0,), dtype=torch.long, device=boxes.device)

    order  = torch.argsort(scores, descending=True)
    boxes  = boxes[order]
    scores = scores[order]
    x = torch.triu(_eiou_matrix(boxes), diagonal=1)

    n      = boxes.shape[0]
    b_prev = torch.ones(n, device=boxes.device, dtype=torch.float32)
    c_final = None
    b_final = b_prev.clone()

    for _ in range(n):
        a_t = torch.diag(b_prev)
        c_t = a_t @ x
        g   = c_t.max(dim=0).values
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
            torch.empty((0,),   device=boxes.device),
            torch.empty((0,),   dtype=torch.long, device=boxes.device),
        )

    c_prime  = (c_final + torch.eye(n, device=boxes.device, dtype=c_final.dtype)) * scores.unsqueeze(1)
    weights  = c_prime[:, keep_mask].transpose(0, 1)
    denom    = weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    merged   = weights @ boxes / denom

    return merged, scores[keep_mask], order[keep_mask]


def _infer_wcnms(model: YOLO, image_path: Path, device) -> list:
    """Run WC-NMS inference at CONF_INFER; return list of (conf, cls_id, [x1,y1,x2,y2])."""
    im, orig_shape = _preprocess_image(image_path, device)

    with torch.no_grad():
        raw_output = model.model(im)

    raw_pred = _unwrap_raw_predictions(raw_output)
    pred     = _decode_raw_predictions(raw_pred, num_classes=len(model.names))

    box_xywh   = pred[:, :4]
    cls_scores = pred[:, 4:]

    if cls_scores.shape[1] < 2:
        raise RuntimeError(f"Model output has fewer than 2 classes: {cls_scores.shape[1]}")

    confs, clses = cls_scores.max(dim=1)

    # WC-NMS operates on bird (0) and drone (1) only; unknown (2) is never predicted
    valid_mask = (clses < 2) & (confs >= CONF_INFER)
    box_xywh   = box_xywh[valid_mask]
    confs      = confs[valid_mask]
    clses      = clses[valid_mask]

    if box_xywh.numel() == 0:
        return []

    box_xyxy = ops.xywh2xyxy(box_xywh)

    if box_xyxy.shape[0] > MAX_DET:
        topk     = torch.argsort(confs, descending=True)[:MAX_DET]
        box_xyxy = box_xyxy[topk]
        confs    = confs[topk]
        clses    = clses[topk]

    preds: list = []
    for class_id in [0, 1]:
        mask = clses == class_id
        if mask.sum() == 0:
            continue

        kept_boxes, kept_scores, _ = _wc_nms_single_class(
            box_xyxy[mask], confs[mask], NMS_THRESH
        )
        if kept_boxes.numel() == 0:
            continue

        kept_boxes = ops.scale_boxes(
            img1_shape=im.shape[2:],
            boxes=kept_boxes.clone(),
            img0_shape=orig_shape,
        )
        for b, s in zip(kept_boxes.cpu(), kept_scores.cpu()):
            preds.append((float(s.item()), int(class_id), [float(v) for v in b.tolist()]))

    return preds


# ---------------------------------------------------------------------------
# Prediction caching
# ---------------------------------------------------------------------------

def _pred_cache_path(model_name: str) -> Path:
    cache_dir = DETECT_DIR / model_name / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"pred_cache_wcnms_{model_name}_{DATASET_SPLIT}_imgsz{IMGSZ}.json"


def _build_pred_cache(
    model_path: Path,
    image_paths: list,
    label_paths: list,
    cache_path: Path,
) -> list:
    print(f"  Loading model weights: {model_path.name}")
    model  = YOLO(str(model_path))
    model.model.eval()
    device = next(model.model.parameters()).device

    total  = len(image_paths)
    cached: list = []
    t0 = time.time()

    for idx, (ip, lp) in enumerate(zip(image_paths, label_paths), 1):
        gt_boxes_xywh, gt_labels = load_label_file(lp)
        with Image.open(ip) as img:
            width, height = img.size
        gt_boxes_xyxy = [xywhn_to_xyxy(b, width, height) for b in gt_boxes_xywh]

        preds = _infer_wcnms(model, ip, device)
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
    return cache_dir / f"sweep_cache_wcnms_{model_name}_{DATASET_SPLIT}_{tag}.npz"


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
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    plot_R = _gaussian_smooth(macro_R, smooth_sigma)
    plot_P = _gaussian_smooth(macro_P, smooth_sigma)

    points   = np.column_stack([plot_R, plot_P]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=float(conf_values.min()), vmax=float(conf_values.max()))
    lc = LineCollection(segments, cmap=PLOT_COLORMAP, norm=norm,
                        linewidth=PLOT_LINEWIDTH, alpha=1.0, zorder=3)
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
    ax.grid(True, alpha=PLOT_GRID_ALPHA, linestyle=PLOT_GRID_LINESTYLE, linewidth=PLOT_GRID_LINEWIDTH)
    ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
    fig.tight_layout()

    out_dir = DETECT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        out = out_dir / f"{model_name}_wc-nms_pr_curve_aod4{ext}"
        fig.savefig(out, dpi=PLOT_DPI)
        print(f"  Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WC-NMS PR curves for YOLO models")
    parser.add_argument("--model",    type=str,   default=None,  help="Process only this model name")
    parser.add_argument("--steps",    type=int,   default=2000,  help="Confidence sweep steps")
    parser.add_argument("--conf-min", type=float, default=0.001, help="Lower conf bound")
    parser.add_argument("--conf-max", type=float, default=0.999, help="Upper conf bound")
    parser.add_argument("--smooth",   type=float, default=PLOT_SMOOTH_SIGMA,
                        help="Gaussian smoothing sigma (0 = off)")
    args = parser.parse_args()

    conf_values = np.linspace(args.conf_min, args.conf_max, args.steps)

    print("=" * 70)
    print("  WC-NMS Precision-Recall Curve Generator")
    print("=" * 70)
    print(f"  Dataset     : {DATASET_SPLIT}  ({IMAGES_DIR})")
    print(f"  Conf sweep  : [{args.conf_min:.3f}, {args.conf_max:.3f}]  x  {args.steps} steps")
    print(f"  IOU thresh  : {IOU_THRESH}")
    print(f"  NMS thresh  : {NMS_THRESH}")

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
            print(f"  Star-marker conf (MODEL_CONF_THRESH): {best_conf:.4f}")
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
