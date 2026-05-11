"""
Video inference: Weighted Boxes Fusion  →  Kalman + LSTM multi-object tracking.

Pipeline per frame:
  1. All ensemble YOLO models predict on the frame.
  2. Detections are clustered and merged with Weighted Boxes Fusion (WBF).
  3. Fused detections feed a greedy-IoU multi-object tracker.
  4. Each track is smoothed by a Kalman filter (constant-velocity, 8-D state).
  5. An LSTM predicts the next box centre from the last SEQ_LEN track positions.

Visual output per track:
  • Coloured bounding box  (Kalman-smoothed)
  • Label  ─  class + track ID
  • Fading trail  (past TRAIL_LEN centres)
  • LSTM predicted next centre  (hollow circle, same colour)
"""

from collections import defaultdict
from pathlib import Path
import csv
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import questionary
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["bird", "drone", "unknown"]
VIDEO_DIR   = PROJECT_ROOT / "dataset" / "test" / "videos"
IMG_SIZE    = 640
DEVICE      = ""             # "cpu", "0", etc.; empty = auto

# Per-model confidence thresholds (tuned on validation set).
MODEL_CONF_THRESH = {
    "yolo8n":  0.6863484706628682,
    "yolo8m":  0.7133918823950539,
    "yolo9t":  0.6724046133517759,
    "yolo10n": 0.5910035105688879,
    "yolo11n": 0.712997868833143,
    "yolo12n": 0.6838702654977842,
    "yolo26n": 0.6052508580184951,
    "fasterrcnn": 0.9891891891891892,
}
CONF_THRESH = 0.70  # fallback if model not in MODEL_CONF_THRESH

COLORS = {
    "bird":    (0, 255, 0),      # green
    "drone":   (0, 0, 255),      # red
    "unknown": (0, 165, 255),    # orange
}

# ── WBF parameters ────────────────────────────────────────────────────────────
FUSION_IOU_THRESH       = 0.50
MIN_MODEL_SUPPORT       = 5
KNOWN_FUSED_CONF_THRESH = 0.68770202403814
SCORE_MARGIN_THRESH     = 0.5582857958051067
DISAGREEMENT_RATIO_THRESH = 0.15223066760019896

# ── Model selection ───────────────────────────────────────────────────────────
# Set True to include a model in the ensemble, False to skip it.
ENABLED_MODELS = {
    "yolo8n":  False,
    "yolo8m":  True,
    "yolo9t":  False,
    "yolo10n": False,
    "yolo11n": True,
    "yolo12n": False,
    "yolo26n": True,
    "fasterrcnn": False,
}

# Ensemble model list: (name, path_to_weights)
MODELS = [
    ("yolo8n",  PROJECT_ROOT / "runs" / "detect" / "yolo8n"  / "weights" / "best.pt"),
    ("yolo8m",  PROJECT_ROOT / "runs" / "detect" / "yolo8m"  / "weights" / "best.pt"),
    ("yolo9t",  PROJECT_ROOT / "runs" / "detect" / "yolo9t"  / "weights" / "best.pt"),
    ("yolo10n", PROJECT_ROOT / "runs" / "detect" / "yolo10n" / "weights" / "best.pt"),
    ("yolo11n", PROJECT_ROOT / "runs" / "detect" / "yolo11n" / "weights" / "best.pt"),
    ("yolo12n", PROJECT_ROOT / "runs" / "detect" / "yolo12n" / "weights" / "best.pt"),
    ("yolo26n", PROJECT_ROOT / "runs" / "detect" / "yolo26n" / "weights" / "best.pt"),
    ("fasterrcnn", PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "fasterrcnn_epoch_50.pt"),
]

# Per-model per-class weighting for WBF (tuned via Bayesian optimisation on 6-model ensemble).
MODEL_WEIGHTS = {
    "yolo8n":  {"bird": 1.0173818474125131, "drone": 1.3684375247874139, "unknown": 1.1646555286393707},
    "yolo8m":  {"bird": 1.000,              "drone": 1.000,              "unknown": 1.000},
    "yolo9t":  {"bird": 1.3345783180967155, "drone": 1.3246461700191348, "unknown": 1.4472517946232146},
    "yolo10n": {"bird": 0.701091472694561,  "drone": 0.9099028240616815, "unknown": 0.718502456356537},
    "yolo11n": {"bird": 1.3215138210731259, "drone": 1.9403016932408828, "unknown": 1.338899849846067},
    "yolo12n": {"bird": 0.8035608928262429, "drone": 1.3438739986019623, "unknown": 1.3166024543945136},
    "yolo26n": {"bird": 1.3364625610235248, "drone": 1.1804482074749882, "unknown": 1.746577505269269},
    "fasterrcnn": {"bird": 1.0, "drone": 1.0, "unknown": 1.0},
}

# ── Tracking parameters ───────────────────────────────────────────────────────
MAX_LOST         = 10    # frames to keep a lost track alive
MIN_HITS         = 2     # frames before a new track is drawn
IOU_THRESH_TRACK = 0.30  # greedy matching IoU threshold
SEQ_LEN          = 8     # LSTM history window (frames)
TRAIL_LEN        = 30    # trail length in frames

# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN wrapper
# ─────────────────────────────────────────────────────────────────────────────
class _FasterRCNNBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy
        self.cls  = cls
        self.conf = conf
    def __len__(self):
        return self.xyxy.shape[0]


class _FasterRCNNResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FasterRCNNWrapper:
    """Wraps a torchvision Faster R-CNN to match the Ultralytics .predict() interface.
    Accepts either a BGR numpy frame (from OpenCV) or a file path string.
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def predict(self, source, conf, imgsz=None, device=None, verbose=False, **kwargs):
        if isinstance(source, np.ndarray):
            # BGR numpy frame → RGB tensor
            rgb = source[:, :, ::-1]
            img_tensor = (
                torch.from_numpy(np.ascontiguousarray(rgb, dtype="uint8"))
                .permute(2, 0, 1)
                .float() / 255.0
            ).to(self.device)
        else:
            img = Image.open(source).convert("RGB")
            img_tensor = (
                torch.from_numpy(np.array(img, dtype="uint8"))
                .permute(2, 0, 1)
                .float() / 255.0
            ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model([img_tensor])

        output = outputs[0]
        boxes  = output["boxes"].cpu()
        labels = output["labels"].float().cpu()
        scores = output["scores"].cpu()

        mask = scores >= conf
        return [_FasterRCNNResult(_FasterRCNNBoxes(boxes[mask], labels[mask], scores[mask]))]


# ─────────────────────────────────────────────────────────────────────────────
# WBF helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_class_name(name: str) -> str:
    lower = str(name).strip().lower()
    if "bird"  in lower: return "bird"
    if "drone" in lower: return "drone"
    return "unknown"


def build_model_class_map(model) -> dict[int, str]:
    names = getattr(model, "names", None)
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): normalize_class_name(v) for k, v in names.items()}
    return {i: normalize_class_name(v) for i, v in enumerate(names)}


def class_name_from_id(cls_id: int) -> str:
    return ["bird", "drone", "unknown"][cls_id] if cls_id < 3 else "unknown"


def _iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def _eiou(a: list[float], b: list[float]) -> float:
    """Extended IoU: IoU minus centre-distance, width, and height penalties.
    EIoU = IoU - (cx_A-cx_B)²+(cy_A-cy_B)² / (wc²+hc²)
                - (wA-wB)² / wc²
                - (hA-hB)² / hc²
    """
    iou = _iou(a, b)
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    cx1 = min(a[0], b[0]); cy1 = min(a[1], b[1])
    cx2 = max(a[2], b[2]); cy2 = max(a[3], b[3])
    wc = cx2 - cx1; hc = cy2 - cy1
    center_p = ((ax - bx) ** 2 + (ay - by) ** 2) / (wc ** 2 + hc ** 2 + 1e-9)
    width_p  = (aw - bw) ** 2 / (wc ** 2 + 1e-9)
    height_p = (ah - bh) ** 2 / (hc ** 2 + 1e-9)
    return iou - center_p - width_p - height_p


def cluster_detections(detections: list[dict], iou_thresh: float) -> list[list[dict]]:
    clusters: list[dict] = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        matched = False
        for cluster in clusters:
            if _iou(det["box"], cluster["rep_box"]) >= iou_thresh:
                cluster["items"].append(det)
                boxes = np.array([item["box"] for item in cluster["items"]], dtype=np.float32)
                cluster["rep_box"] = boxes.mean(axis=0).tolist()
                matched = True
                break
        if not matched:
            clusters.append({"items": [det], "rep_box": det["box"][:]})
    return [c["items"] for c in clusters]


def fuse_cluster(cluster_items: list[dict]) -> dict | None:
    class_scores = {0: 0.0, 1: 0.0}
    for det in cluster_items:
        if det["class_id"] in (0, 1):
            class_scores[det["class_id"]] += det["weighted_score"]

    best  = max(class_scores, key=class_scores.__getitem__)
    second = 1 - best
    margin = class_scores[best] - class_scores[second]
    disagreement = class_scores[second] / (class_scores[best] + 1e-12)

    support_models = {d["model"] for d in cluster_items if d["class_id"] == best}
    best_items     = [d for d in cluster_items if d["class_id"] == best]
    fused_conf_best = float(np.mean([d["confidence"] for d in best_items])) if best_items else 0.0

    uncertain = (
        len(support_models) < MIN_MODEL_SUPPORT
        or fused_conf_best < KNOWN_FUSED_CONF_THRESH
        or margin < SCORE_MARGIN_THRESH
        or disagreement > DISAGREEMENT_RATIO_THRESH
    )

    final_class = 2 if uncertain else best
    chosen      = cluster_items if uncertain else best_items
    if not chosen:
        return None

    score_sum = sum(d["weighted_score"] for d in chosen) or 1e-9
    x1 = sum(d["weighted_score"] * d["box"][0] for d in chosen) / score_sum
    y1 = sum(d["weighted_score"] * d["box"][1] for d in chosen) / score_sum
    x2 = sum(d["weighted_score"] * d["box"][2] for d in chosen) / score_sum
    y2 = sum(d["weighted_score"] * d["box"][3] for d in chosen) / score_sum

    return {
        "box":       [float(x1), float(y1), float(x2), float(y2)],
        "class_id":  int(final_class),
        "confidence": float(np.mean([d["confidence"] for d in chosen])),
    }


def run_wbf_on_frame(loaded_models: list, frame: np.ndarray) -> list[dict]:
    """Run all ensemble models on one BGR frame and return fused detections."""
    all_detections: list[dict] = []

    for model_name, model, class_map in loaded_models:
        predict_kwargs = {
            "source": frame,
            "conf":   MODEL_CONF_THRESH.get(model_name, CONF_THRESH),
            "imgsz":  IMG_SIZE,
            "save":   False,
            "show":   False,
            "verbose": False,
        }
        if DEVICE:
            predict_kwargs["device"] = DEVICE

        results = model.predict(**predict_kwargs)
        r = results[0] if isinstance(results, list) else results

        if not hasattr(r, "boxes") or len(r.boxes) == 0:
            continue

        boxes_xyxy  = r.boxes.xyxy.cpu().numpy()
        class_ids   = r.boxes.cls.cpu().numpy().astype(int)
        confidences = r.boxes.conf.cpu().numpy().astype(float)

        for box, cls_id, conf in zip(boxes_xyxy, class_ids, confidences):
            cls_name   = class_map.get(int(cls_id), class_name_from_id(int(cls_id)))
            cls_idx    = 0 if cls_name == "bird" else 1 if cls_name == "drone" else 2
            mw         = MODEL_WEIGHTS.get(model_name, {}).get(cls_name, 1.0)
            all_detections.append({
                "box":           [float(v) for v in box.tolist()],
                "class_id":      cls_idx,
                "confidence":    float(conf),
                "model":         model_name,
                "weighted_score": float(mw * conf),
            })

    if not all_detections:
        return []

    fused: list[dict] = []
    for cluster in cluster_detections(all_detections, FUSION_IOU_THRESH):
        item = fuse_cluster(cluster)
        if item is not None:
            fused.append(item)

    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter
# ─────────────────────────────────────────────────────────────────────────────
class KalmanBoxTracker:
    """Constant-velocity Kalman filter.  State: [cx, cy, w, h, vcx, vcy, vw, vh]"""

    _next_id = 0

    def __init__(self, xywh: list[float]):
        cx, cy, w, h = xywh
        self.id   = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.hits = 1
        self.lost = 0
        self.history: list[list[float]] = [[cx, cy, w, h]]

        dt = 1.0
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt

        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        self.R = np.eye(4, dtype=np.float32) * 4.0
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        self.P = np.eye(8, dtype=np.float32)
        self.P[4:, 4:] *= 1000.0
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].flatten()

    def update(self, xywh: list[float]):
        z = np.array(xywh, dtype=np.float32).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P  = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        self.hits += 1
        self.lost  = 0
        self.history.append(list(xywh))

    def get_state(self) -> np.ndarray:
        return self.x[:4].flatten()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM trajectory predictor
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryLSTM(nn.Module):
    """Predicts next normalised [cx, cy, w, h] from SEQ_LEN past boxes."""

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywh_to_xyxy(box) -> list[float]:
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-object tracker
# ─────────────────────────────────────────────────────────────────────────────
class MultiObjectTracker:
    def __init__(self, lstm_device: torch.device):
        self.tracks: list[tuple[KalmanBoxTracker, int, float, object]] = []
        self.lstm   = TrajectoryLSTM().to(lstm_device).eval()
        self.device = lstm_device
        self.trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def update(
        self,
        detections: list[tuple[list[float], int, float]],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[list[float], int, int, float, list[float] | None]]:

        # Kalman predict
        preds_xyxy = [xywh_to_xyxy(trk.predict()) for trk, _, _c, __ in self.tracks]

        # Greedy IoU matching
        n_t, n_d = len(preds_xyxy), len(detections)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        pairs: list[tuple[int, int]] = []

        if n_t and n_d:
            cost = np.zeros((n_t, n_d), dtype=np.float32)
            for ti, pxy in enumerate(preds_xyxy):
                for di, (dxy, _, _c) in enumerate(detections):
                    cost[ti, di] = _iou(pxy, dxy)

            for val, ti, di in sorted(
                ((cost[ti, di], ti, di) for ti in range(n_t) for di in range(n_d)),
                reverse=True,
            ):
                if val < IOU_THRESH_TRACK:
                    break
                if ti not in matched_t and di not in matched_d:
                    pairs.append((ti, di))
                    matched_t.add(ti); matched_d.add(di)

        # Update matched
        for ti, di in pairs:
            dxy, dcls, dconf = detections[di]
            xywh = xyxy_to_xywh(dxy)
            trk, _, _c, hidden = self.tracks[ti]
            trk.update(xywh)
            hidden = self._lstm_step(xywh, frame_w, frame_h, hidden)
            self.tracks[ti] = (trk, dcls, dconf, hidden)

        # Increment lost for unmatched tracks
        for ti, (trk, cls, conf, hid) in enumerate(self.tracks):
            if ti not in matched_t:
                trk.lost += 1

        # Spawn new tracks
        for di, (dxy, dcls, dconf) in enumerate(detections):
            if di not in matched_d:
                self.tracks.append((KalmanBoxTracker(xyxy_to_xywh(dxy)), dcls, dconf, None))

        # Prune dead tracks
        self.tracks = [(t, c, cf, h) for t, c, cf, h in self.tracks if t.lost <= MAX_LOST]

        # Build output + update trails
        results: list[tuple[list[float], int, int, float, list[float] | None]] = []
        for trk, cls, conf, hidden in self.tracks:
            if trk.hits < MIN_HITS and trk.lost > 0:
                continue
            state    = trk.get_state()
            box_xyxy = xywh_to_xyxy(state)
            cx, cy   = int(state[0]), int(state[1])

            trail = self.trails[trk.id]
            trail.append((cx, cy))
            if len(trail) > TRAIL_LEN:
                del trail[:-TRAIL_LEN]

            lstm_pred = self._lstm_predict(trk.history, frame_w, frame_h)
            results.append((box_xyxy, trk.id, cls, conf, lstm_pred))

        live_ids = {trk.id for trk, _, _c, __ in self.tracks}
        for tid in [k for k in self.trails if k not in live_ids]:
            del self.trails[tid]

        return results

    def _lstm_step(self, xywh, fw, fh, hidden):
        norm = [[xywh[0] / fw, xywh[1] / fh, xywh[2] / fw, xywh[3] / fh]]
        x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, hidden = self.lstm(x, hidden)
        return hidden

    def _lstm_predict(self, history, fw, fh) -> list[float] | None:
        if len(history) < SEQ_LEN:
            return None
        seq = [[b[0] / fw, b[1] / fh, b[2] / fw, b[3] / fh]
               for b in history[-SEQ_LEN:]]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred, _ = self.lstm(x, None)
        p = pred[0].cpu().numpy()
        return xywh_to_xyxy(
            [float(p[0]) * fw, float(p[1]) * fh,
             float(p[2]) * fw, float(p[3]) * fh]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_tracks(frame: np.ndarray, track_results: list, trails: dict):
    for box_xyxy, track_id, cls_int, conf, lstm_pred in track_results:
        label = CLASS_NAMES[cls_int] if cls_int < len(CLASS_NAMES) else "unknown"
        color = COLORS[label]

        x1, y1, x2, y2 = (int(v) for v in box_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_bg_y1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, text_bg_y1), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame, text, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

        trail = trails.get(track_id, [])
        n = len(trail)
        for i in range(1, n):
            alpha = i / n
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, trail[i - 1], trail[i], c, 2)

        if lstm_pred is not None:
            lx1, ly1, lx2, ly2 = lstm_pred
            lcx = int((lx1 + lx2) / 2)
            lcy = int((ly1 + ly2) / 2)
            cv2.circle(frame, (lcx, lcy), 6, color, 2)


# ─────────────────────────────────────────────────────────────────────────────
# GT label helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_gt_labels(video_path: Path) -> tuple[list, list] | None:
    """Load <stem>.json GT alongside the video if present.

    Expected JSON keys:
      exist   – list[int]  1 = target present, 0 = absent
      gt_rect – list[[x, y, w, h]]  top-left pixel coords
    """
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get("exist", []), data.get("gt_rect", [])


# ─────────────────────────────────────────────────────────────────────────────
# Questionary helpers
# ─────────────────────────────────────────────────────────────────────────────
def choose_video_file() -> Path:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_paths:
        raise ValueError(f"No .mp4 video files found in {VIDEO_DIR}")

    if len(sys.argv) > 1:
        video_name = sys.argv[1]
        if not video_name.lower().endswith(".mp4"):
            video_name = f"{video_name}.mp4"
        video_path = VIDEO_DIR / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path

    if len(video_paths) == 1:
        return video_paths[0]

    video_name = questionary.select(
        "Choose a video from dataset/test/videos:",
        choices=[p.name for p in video_paths],
    ).ask()
    if not video_name:
        raise ValueError("No video selected")
    return VIDEO_DIR / video_name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    video_path = choose_video_file()

    output_dir = PROJECT_ROOT / "runs" / "detect" / "inference_video"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_wbf_tracking.mp4"

    # Torch device for LSTM
    if DEVICE:
        lstm_device = torch.device("cuda:0" if DEVICE == "0" else DEVICE)
    else:
        lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ensemble models
    print("Loading ensemble models...")
    loaded_models: list[tuple[str, object, dict]] = []
    for model_name, model_path in MODELS:
        if not ENABLED_MODELS.get(model_name, False):
            continue
        if not model_path.exists():
            print(f"  [SKIP] {model_name}: not found at {model_path}")
            continue
        if model_name == "fasterrcnn":
            _frcnn_device = torch.device(
                "cuda" if torch.cuda.is_available() and DEVICE != "cpu" else "cpu"
            )
            _ckpt = torch.load(str(model_path), map_location=_frcnn_device, weights_only=False)
            _num_classes = _ckpt["model_state_dict"][
                "roi_heads.box_predictor.cls_score.weight"
            ].shape[0]
            _frcnn = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            _in_features = _frcnn.roi_heads.box_predictor.cls_score.in_features
            _frcnn.roi_heads.box_predictor = FastRCNNPredictor(_in_features, _num_classes)
            _frcnn.load_state_dict(_ckpt["model_state_dict"])
            _frcnn.to(_frcnn_device).eval()
            model     = FasterRCNNWrapper(_frcnn, _frcnn_device)
            class_map = {1: "bird", 2: "drone", 3: "unknown"}
        else:
            model     = YOLO(str(model_path))
            class_map = build_model_class_map(model)
        loaded_models.append((model_name, model, class_map))
        print(f"  [OK]   {model_name}")

    if not loaded_models:
        raise ValueError("No ensemble models were loaded. Check MODELS paths.")

    global MIN_MODEL_SUPPORT
    majority = (len(loaded_models) + 1) // 2  # ceil(n/2)
    if MIN_MODEL_SUPPORT > len(loaded_models) or MIN_MODEL_SUPPORT > majority:
        new_val = min(MIN_MODEL_SUPPORT, majority)
        new_val = max(1, new_val)
        print(
            f"  [NOTE] MIN_MODEL_SUPPORT adjusted from {MIN_MODEL_SUPPORT} "
            f"→ majority threshold {majority} for {len(loaded_models)} loaded models."
        )
        MIN_MODEL_SUPPORT = majority

    tracker = MultiObjectTracker(lstm_device=lstm_device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(f"\n{'Input video:':<22}{video_path}")
    print(f"{'Saving output to:':<22}{output_path}")

    # Ground-truth labels (optional)
    gt_data    = load_gt_labels(video_path)
    has_gt     = gt_data is not None
    gt_exist: list = []
    gt_rect:  list = []
    if has_gt:
        gt_exist, gt_rect = gt_data
        print(f"{'GT labels:':<22}{video_path.stem}.json  ({len(gt_exist)} frames)")

    print()

    # Per-frame evaluation accumulators
    frame_ious:     list[float] = []  # IoU vs GT per exist=1 frame
    frame_eious:    list[float] = []  # EIoU vs GT per exist=1 frame
    frame_dists:    list[float] = []  # centre distance per exist=1 frame
    frame_confs:    list[float] = []  # best track confidence per exist=1 frame
    frame_cls_ids:  list[int]   = []  # class ID of best-matching track per exist=1 frame (-1 = none)
    frame_numbers:  list[int]   = []  # 1-based frame index
    map_preds:      list[tuple[float, float, int]] = []  # (conf, iou_with_gt, frame_idx) for all preds in exist=1 frames
    covered_frames: int = 0           # exist=1 frames where ≥1 track present
    exist1_frames:  int = 0           # total exist=1 frames seen

    processed  = 0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Stage 1 — WBF fusion
        fused = run_wbf_on_frame(loaded_models, frame)

        # Stage 2 — Kalman + LSTM tracking
        detections = [(d["box"], d["class_id"], d["confidence"]) for d in fused]
        track_results = tracker.update(detections, width, height)

        draw_tracks(frame, track_results, tracker.trails)

        # ── GT overlay + per-frame evaluation ───────────────────────────────
        if has_gt and processed < len(gt_exist):
            if gt_exist[processed] == 1:
                rx, ry, rw, rh = gt_rect[processed]
                gx1, gy1 = int(rx), int(ry)
                gx2, gy2 = int(rx + rw), int(ry + rh)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 255, 255), 2)
                cv2.putText(
                    frame, "GT", (gx1, max(12, gy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                )

                gt_xyxy = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
                gt_cx   = rx + rw / 2.0
                gt_cy   = ry + rh / 2.0
                exist1_frames += 1

                best_iou  = 0.0
                best_eiou = 0.0
                best_dist = float("nan")
                best_conf = 0.0
                best_cls  = -1
                for box_xyxy, _tid, _cls, _conf, _lstm in track_results:
                    iou_val = _iou(box_xyxy, gt_xyxy)
                    if iou_val > best_iou:
                        best_iou  = iou_val
                        best_eiou = _eiou(box_xyxy, gt_xyxy)
                        tcx = (box_xyxy[0] + box_xyxy[2]) / 2.0
                        tcy = (box_xyxy[1] + box_xyxy[3]) / 2.0
                        best_dist = float(np.hypot(tcx - gt_cx, tcy - gt_cy))
                        best_cls  = int(_cls)
                    if float(_conf) > best_conf:
                        best_conf = float(_conf)
                    map_preds.append((float(_conf), iou_val, processed))

                if track_results:
                    covered_frames += 1

                frame_ious.append(best_iou)
                frame_eious.append(best_eiou)
                frame_dists.append(best_dist)
                frame_confs.append(best_conf)
                frame_cls_ids.append(best_cls)
                frame_numbers.append(processed + 1)

        writer.write(frame)

        processed += 1
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)

        if total_frames > 0:
            pct = processed / total_frames * 100
            print(
                f"Progress: {processed}/{total_frames} ({pct:.2f}%)"
                f" Elapsed: {minutes}:{seconds:02d}",
                end="\r",
            )
        else:
            print(
                f"Progress: {processed} frames Elapsed: {minutes}:{seconds:02d}",
                end="\r",
            )

    cap.release()
    writer.release()

    print()
    if total_frames > 0:
        print(f"Done. Processed {processed}/{total_frames} frames.")
    else:
        print(f"Done. Processed {processed} frames.")

    # ── Evaluation summary ───────────────────────────────────────────────────
    if has_gt and exist1_frames > 0:
        iou_arr  = np.array(frame_ious,  dtype=np.float32)
        dist_arr = np.array(frame_dists, dtype=np.float32)

        thr_iou  = np.linspace(0.0, 1.0, 101)
        success  = np.array([(iou_arr >= t).mean() for t in thr_iou], dtype=np.float32)
        auc      = float(np.trapezoid(success, thr_iou))
        sr50     = float((iou_arr >= 0.5).mean())

        thr_dist = np.arange(0, 51, dtype=np.float32)
        precision = np.array([(dist_arr <= t).mean() for t in thr_dist], dtype=np.float32)
        prec20   = float((dist_arr <= 20.0).mean())
        coverage = covered_frames / exist1_frames
        mean_iou = float(iou_arr.mean())

        # ── mAP (COCO-style 101-point interpolation) ─────────────────────
        def _compute_ap(preds: list, gt_count: int, iou_thresh: float) -> float:
            """AP with greedy matching (one TP per GT box per frame)."""
            if not preds or gt_count == 0:
                return 0.0
            sorted_preds = sorted(preds, key=lambda x: x[0], reverse=True)
            matched: set[int] = set()
            tp_list: list[int] = []
            for _conf, iou, fidx in sorted_preds:
                if iou >= iou_thresh and fidx not in matched:
                    tp_list.append(1)
                    matched.add(fidx)
                else:
                    tp_list.append(0)
            cum_tp   = np.cumsum(tp_list, dtype=np.float64)
            cum_fp   = np.cumsum([1 - t for t in tp_list], dtype=np.float64)
            recall    = cum_tp / gt_count
            precision_vals = cum_tp / (cum_tp + cum_fp + 1e-9)
            # 101-point interpolated AP (COCO style)
            ap = 0.0
            for thr in np.linspace(0.0, 1.0, 101):
                mask = recall >= thr
                ap  += (precision_vals[mask].max() if mask.any() else 0.0)
            return ap / 101

        ap50      = _compute_ap(map_preds, exist1_frames, 0.50)
        map50_95  = float(np.mean([
            _compute_ap(map_preds, exist1_frames, t)
            for t in np.arange(0.50, 1.00, 0.05)
        ]))

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  Evaluation  ({exist1_frames} present frames / {processed} total)")
        print(sep)
        print(f"  {'Mean IoU:':<32}{mean_iou * 100:.2f}%")
        print(f"  {'Success Rate  @IoU≥0.5:':<32}{sr50 * 100:.2f}%")
        print(f"  {'AUC  (success curve 0→1):':<32}{auc * 100:.2f}%")
        print(f"  {'Precision  @20 px:':<32}{prec20 * 100:.2f}%")
        print(f"  {'Coverage:':<32}{coverage * 100:.2f}%  ({covered_frames}/{exist1_frames})")
        print(f"  {'AP@50:':<32}{ap50 * 100:.2f}%")
        print(f"  {'mAP@50:95:':<32}{map50_95 * 100:.2f}%")
        print(sep)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(thr_iou, success, linewidth=2)
        ax1.set_xlabel("IoU threshold")
        ax1.set_ylabel("Success rate")
        ax1.set_title(f"Success curve  AUC={auc:.3f}")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.4)

        ax2.plot(thr_dist, precision, linewidth=2)
        ax2.axvline(20, color="gray", linestyle="--", linewidth=1, label="20 px")
        ax2.set_xlabel("Centre distance threshold (px)")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"Precision curve  @20px={prec20:.3f}")
        ax2.set_xlim(0, 50); ax2.set_ylim(0, 1)
        ax2.legend(); ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        plot_path = output_dir / f"{video_path.stem}_wbf_eval.png"
        fig.savefig(str(plot_path), dpi=120)
        plt.close(fig)
        print(f"\n  Eval plot → {plot_path}")

        # ── Per-frame metrics plot ──────────────────────────────────────────
        frames_x  = np.array(frame_numbers, dtype=np.float32)
        iou_vals  = np.array(frame_ious,   dtype=np.float32)
        eiou_vals = np.array(frame_eious,  dtype=np.float32)
        # EIoU penalty = IoU − EIoU: shows centre/shape mismatch, always ≥ 0.
        # (EIoU ≈ IoU when tracking is tight; penalty spikes on alignment failures.)
        eiou_penalty = iou_vals - eiou_vals
        metrics = [
            ("IoU",                     iou_vals,                                  (0.0,  1.0)),
            ("EIoU penalty (IoU−EIoU)", eiou_penalty,                              (0.0,  None)),
            ("Center distance (px)",    np.array(frame_dists,   dtype=np.float32), None),
            ("Detection confidence",    np.array(frame_confs,   dtype=np.float32), (0.0,  1.0)),
            ("Class ID",                np.array(frame_cls_ids, dtype=np.float32), (-1.5, 2.5)),
        ]

        fig2, axes = plt.subplots(
            len(metrics), 1,
            figsize=(12, 3 * len(metrics)),
            sharex=True,
        )
        fig2.suptitle("WBF tracking  —  per-frame metrics", fontsize=13, fontweight="bold")

        for ax, (label, values, ylim) in zip(axes, metrics):
            ax.plot(frames_x, values, linewidth=1.0)
            ax.set_ylabel(label, fontsize=10)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.35)

        # Class ID axis: integer ticks with class name labels (-1 = none)
        cls_ax = axes[-1]
        cls_ax.set_yticks([-1, 0, 1, 2])
        cls_ax.set_yticklabels(["none", "bird", "drone", "unknown"], fontsize=8)

        axes[-1].set_xlabel("Frame", fontsize=10)
        fig2.tight_layout()
        perframe_plot_path = output_dir / f"{video_path.stem}_wbf_perframe.png"
        fig2.savefig(str(perframe_plot_path), dpi=120)
        plt.close(fig2)
        print(f"  Per-frame plot → {perframe_plot_path}")

        npz_path = output_dir / f"{video_path.stem}_wbf_perframe.npz"
        np.savez(
            str(npz_path),
            frame_numbers=np.array(frame_numbers, dtype=np.int32),
            frame_ious=np.array(frame_ious, dtype=np.float32),
            frame_eious=np.array(frame_eious, dtype=np.float32),
            frame_dists=np.array(frame_dists, dtype=np.float32),
            frame_confs=np.array(frame_confs, dtype=np.float32),
            frame_cls_ids=np.array(frame_cls_ids, dtype=np.int32),
        )
        print(f"  Per-frame npz  → {npz_path}")

        csv_path = output_dir / f"{video_path.stem}_wbf_eval.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value", "value_percent", "note"])
            writer.writerow(["mean_iou", mean_iou, mean_iou * 100.0, ""])
            writer.writerow(["success_rate_iou_ge_0_5", sr50, sr50 * 100.0, "IoU >= 0.5"])
            writer.writerow(["auc_success_curve_0_to_1", auc, auc * 100.0, ""])
            writer.writerow(["precision_at_20px", prec20, prec20 * 100.0, "center distance <= 20"])
            writer.writerow(["coverage", coverage, coverage * 100.0, f"{covered_frames}/{exist1_frames}"])
            writer.writerow(["ap50", ap50, ap50 * 100.0, "COCO-style AP @ IoU=0.50"])
            writer.writerow(["map50_95", map50_95, map50_95 * 100.0, "COCO-style mAP @ IoU=0.50:0.95"])
            writer.writerow(["covered_frames", covered_frames, "", ""])
            writer.writerow(["present_frames", exist1_frames, "", ""])
            writer.writerow(["processed_frames", processed, "", ""])
        print(f"  Eval csv  → {csv_path}")


if __name__ == "__main__":
    main()
