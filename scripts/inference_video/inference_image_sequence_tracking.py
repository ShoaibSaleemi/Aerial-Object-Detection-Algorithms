"""
Video inference with multi-object tracking.

Pipeline per frame:
  1. YOLO detection
  2. Greedy IoU matching  →  associates detections to existing tracks
  3. Kalman filter        →  smooths state, predicts position when lost
  4. LSTM predictor       →  forecasts next centre from track history
                             (shown as a hollow circle on the frame)

Each track renders:
  • Coloured bounding box  (Kalman-smoothed)
  • Label  ─  class + track ID
  • Fading trail  (past TRAIL_LEN centres)
  • LSTM predicted next centre  (hollow circle, same colour)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import sys
import time
import rtamt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import questionary
import torch
import torch.nn as nn
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["bird", "drone", "unknown"]
IMAGES_DIR  = PROJECT_ROOT / "dataset 2" / "test" / "images"
LABELS_DIR  = PROJECT_ROOT / "dataset 2" / "test" / "labels"
import re
import glob
CONF_THRESH = 0.7
IMG_SIZE    = 640
DEVICE      = ""  # "cpu", "0", etc.; empty = auto.

COLORS = {
    "bird":    (0, 255, 0),      # green
    "drone":   (0, 0, 255),      # red
    "unknown": (0, 165, 255),    # orange
}

# ── Tracking hyper-parameters ────────────────────────────────────────────────
MAX_LOST  = 10   # frames before a lost track is deleted
MIN_HITS  = 2    # frames before a new track is displayed
IOU_THRESH = 0.30  # minimum IoU to match a detection to a track
SEQ_LEN   = 8    # LSTM input sequence length (frames of history)
TRAIL_LEN = 30   # number of past centres drawn as a trail

# ── WBF ensemble parameters ───────────────────────────────────────────────────
FUSION_IOU_THRESH         = 0.50
MIN_MODEL_SUPPORT         = 5
KNOWN_FUSED_CONF_THRESH   = 0.68770202403814
SCORE_MARGIN_THRESH       = 0.5582857958051067
DISAGREEMENT_RATIO_THRESH = 0.15223066760019896
FA_IOU_THRESH = 0.5  # IoU threshold: drone-present frames where track IoU < this are FAs

MODEL_CONF_THRESH: dict[str, float] = {
    "yolo8n":  0.6863484706628682,
    "yolo8m":  0.7133918823950539,
    "yolo9t":  0.6724046133517759,
    "yolo10n": 0.5910035105688879,
    "yolo11n": 0.712997868833143,
    "yolo12n": 0.6838702654977842,
    "yolo26n": 0.6052508580184951,
}

# Set True to include a model in the WBF ensemble, False to skip it.
ENABLED_MODELS: dict[str, bool] = {
    "yolo8n":  False,
    "yolo8m":  True,
    "yolo9t":  False,
    "yolo10n": False,
    "yolo11n": True,
    "yolo12n": False,
    "yolo26n": True,
}

MODELS: list[tuple[str, Path]] = [
    ("yolo8n",  PROJECT_ROOT / "runs" / "detect" / "yolo8n"  / "weights" / "best.pt"),
    ("yolo8m",  PROJECT_ROOT / "runs" / "detect" / "yolo8m"  / "weights" / "best.pt"),
    ("yolo9t",  PROJECT_ROOT / "runs" / "detect" / "yolo9t"  / "weights" / "best.pt"),
    ("yolo10n", PROJECT_ROOT / "runs" / "detect" / "yolo10n" / "weights" / "best.pt"),
    ("yolo11n", PROJECT_ROOT / "runs" / "detect" / "yolo11n" / "weights" / "best.pt"),
    ("yolo12n", PROJECT_ROOT / "runs" / "detect" / "yolo12n" / "weights" / "best.pt"),
    ("yolo26n", PROJECT_ROOT / "runs" / "detect" / "yolo26n" / "weights" / "best.pt"),
]

MODEL_WEIGHTS: dict[str, dict[str, float]] = {
    "yolo8n":  {"bird": 1.0173818474125131, "drone": 1.3684375247874139, "unknown": 1.1646555286393707},
    "yolo8m":  {"bird": 1.0,                "drone": 1.0,                "unknown": 1.0},
    "yolo9t":  {"bird": 1.3345783180967155, "drone": 1.3246461700191348, "unknown": 1.4472517946232146},
    "yolo10n": {"bird": 0.701091472694561,  "drone": 0.9099028240616815, "unknown": 0.718502456356537},
    "yolo11n": {"bird": 1.3215138210731259, "drone": 1.9403016932408828, "unknown": 1.338899849846067},
    "yolo12n": {"bird": 0.8035608928262429, "drone": 1.3438739986019623, "unknown": 1.3166024543945136},
    "yolo26n": {"bird": 1.3364625610235248, "drone": 1.1804482074749882, "unknown": 1.746577505269269},
}

# ── STL / RTM requirement thresholds ─────────────────────────────────────────
# REQ-01: flash rate [Hz] + red-channel dominance
REQ01_FLASH_HZ_LO   = 1.0     # lower flash-rate bound (Hz)  — was 40; unreachable at 30fps
REQ01_FLASH_HZ_HI   = 14.0    # upper flash-rate bound (Hz)  — Nyquist limit ≈ fps/2
REQ01_RED_RATIO      = 0.45   # red-channel dominance fraction of max(R,G,B)
REQ01_BRIGHTNESS_WIN = 16     # sliding window (frames) for oscillation counting
# REQ-02: hover / stable aspect-ratio
REQ02_AR_VAR_MAX     = 0.01   # max aspect-ratio variance → "constant AR"
REQ02_VEL_MAX        = 1.5    # max Kalman velocity magnitude (px/frame) → hover
# REQ-03: no-light confidence
# REQ-04: shape deformation
REQ04_SHAPE_EPS      = 999.0  # DISABLED — 0 satisfactions on dataset; only produced violations
REQ04_DEFORM_WIN     = 12     # window (frames) for deformation measurement  — was 8
# REQ-05: kinematic STL classifier (Kalman-derived signals)
REQ05_MIN_HISTORY    = 25     # minimum track frames before evaluation  — was 15
REQ05_HOVER_VEL      = 1.0   # px/frame — speed threshold for hovering  — was 1.5
REQ05_HOVER_VZ       = 0.8   # px/frame — vertical vel threshold for hovering
REQ05_BIRD_MAX_VEL   = 4.0   # px/frame — birds stay below this speed  — was 8.0
REQ05_FAST_VEL       = 6.0   # px/frame — fast motion (helicopter transit)  — was 5.0
REQ05_CRUISE_VEL     = 5.0   # px/frame — sustained cruise speed (airplane)  — was 4.0
REQ05_HOVER_WINDOW   = 10    # frames — temporal window for hover detection
REQ05_HOVER_DURATION = 8     # frames — sustained hover duration  — was 5


# ── Kalman filter ─────────────────────────────────────────────────────────────
class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter for a single bounding box.

    State   : [cx, cy, w, h, vcx, vcy, vw, vh]
    Measure : [cx, cy, w, h]
    """

    _next_id = 0

    def __init__(self, xywh: list[float]):
        cx, cy, w, h = xywh
        self.id   = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.hits = 1
        self.lost = 0
        # Raw measurement history used by the LSTM.
        self.history: list[list[float]] = [[cx, cy, w, h]]

        dt = 1.0
        # Transition matrix (8×8) – constant velocity.
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Observation matrix (4×8).
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        self.R = np.eye(4, dtype=np.float32) * 4.0    # measurement noise
        self.Q = np.eye(8, dtype=np.float32)           # process noise
        self.Q[4:, 4:] *= 0.01
        self.P = np.eye(8, dtype=np.float32)           # state covariance
        self.P[4:, 4:] *= 1000.0

        self.x = np.array(
            [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        ).reshape(8, 1)

    def predict(self) -> np.ndarray:
        """Advance state one step; return predicted [cx, cy, w, h]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].flatten()

    def update(self, xywh: list[float]):
        """Correct state with a new measurement [cx, cy, w, h]."""
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
        """Return current smoothed [cx, cy, w, h]."""
        return self.x[:4].flatten()


# ── LSTM trajectory predictor ─────────────────────────────────────────────────
class TrajectoryLSTM(nn.Module):
    """
    Predicts the next normalised [cx, cy, w, h] from a sequence of past boxes.

    Input : (1, seq_len, 4)  — values normalised by frame width / height
    Output: (1, 4)
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden


# ── Box geometry helpers ──────────────────────────────────────────────────────
def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywh_to_xyxy(box) -> list[float]:
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def compute_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / (ua + ub - inter + 1e-9)


# ── WBF ensemble helpers ──────────────────────────────────────────────────────
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


def cluster_detections(detections: list[dict], iou_thresh: float) -> list[list[dict]]:
    """Greedily cluster detections by IoU similarity (highest-confidence first)."""
    clusters: list[dict] = []
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


def fuse_cluster(cluster_items: list[dict]) -> dict | None:
    """Fuse a cluster of overlapping detections into one box via weighted averaging."""
    # Accumulate weighted scores per known class (0=bird, 1=drone)
    class_scores: dict[int, float] = {0: 0.0, 1: 0.0}
    for det in cluster_items:
        if det["class_id"] in (0, 1):
            class_scores[det["class_id"]] += det["weighted_score"]

    best   = max(class_scores, key=class_scores.__getitem__)
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
        "box":        [float(x1), float(y1), float(x2), float(y2)],
        "class_id":   int(final_class),
        "confidence": float(np.mean([d["confidence"] for d in chosen])),
    }


def run_wbf_on_frame(loaded_models: list, frame: np.ndarray) -> list[dict]:
    """Run all ensemble models on a frame and return WBF-fused detections."""
    all_detections: list[dict] = []
    for model_name, model, class_map in loaded_models:
        predict_kwargs: dict = {
            "source":  frame,
            "conf":    MODEL_CONF_THRESH.get(model_name, CONF_THRESH),
            "imgsz":   IMG_SIZE,
            "save":    False,
            "show":    False,
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
            cls_name = class_map.get(int(cls_id), class_name_from_id(int(cls_id)))
            cls_idx  = 0 if cls_name == "bird" else 1 if cls_name == "drone" else 2
            mw       = MODEL_WEIGHTS.get(model_name, {}).get(cls_name, 1.0)
            all_detections.append({
                "box":            [float(v) for v in box.tolist()],
                "class_id":       cls_idx,
                "confidence":     float(conf),
                "model":          model_name,
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


# ── Multi-object tracker ──────────────────────────────────────────────────────
class MultiObjectTracker:
    """
    Manages the full track lifecycle:
      • greedy IoU matching
      • Kalman predict / update
      • LSTM hidden-state stepping and next-box prediction
      • trail (past centres) accumulation
    """

    def __init__(self, lstm_device: torch.device):
        # Each entry: (KalmanBoxTracker, class_id, confidence, lstm_hidden_state)
        self.tracks: list[tuple[KalmanBoxTracker, int, float, object]] = []
        self.lstm   = TrajectoryLSTM().to(lstm_device).eval()
        self.device = lstm_device
        # track_id → list of (int cx, int cy) pixel centres
        self.trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # ── public ───────────────────────────────────────────────────────────────
    def update(
        self,
        detections: list[tuple[list[float], int, float]],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[list[float], int, int, float, list[float] | None]]:
        """
        Process one frame.

        detections : list of (box_xyxy, class_id)
        Returns    : list of (box_xyxy, track_id, class_id, lstm_pred_xyxy_or_None)
        """
        # ── Kalman predict ──
        preds_xyxy = [xywh_to_xyxy(trk.predict()) for trk, _, _cf, __ in self.tracks]

        # ── Greedy IoU matching ──
        n_t, n_d = len(preds_xyxy), len(detections)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        pairs: list[tuple[int, int]] = []

        if n_t and n_d:
            cost = np.zeros((n_t, n_d), dtype=np.float32)
            for ti, pxy in enumerate(preds_xyxy):
                for di, (dxy, _, _dc) in enumerate(detections):
                    cost[ti, di] = compute_iou(pxy, dxy)

            # Sort all pairs by descending IoU, pick greedily.
            candidates = sorted(
                ((cost[ti, di], ti, di) for ti in range(n_t) for di in range(n_d)),
                reverse=True,
            )
            for val, ti, di in candidates:
                if val < IOU_THRESH:
                    break
                if ti not in matched_t and di not in matched_d:
                    pairs.append((ti, di))
                    matched_t.add(ti)
                    matched_d.add(di)

        # ── Update matched tracks ──
        for ti, di in pairs:
            dxy, dcls, dconf = detections[di]
            xywh = xyxy_to_xywh(dxy)
            trk, _, _old_conf, hidden = self.tracks[ti]
            trk.update(xywh)
            hidden = self._lstm_step(xywh, frame_w, frame_h, hidden)
            self.tracks[ti] = (trk, dcls, dconf, hidden)

        # ── Increment lost counter for unmatched tracks ──
        for ti, (trk, cls, conf, hid) in enumerate(self.tracks):
            if ti not in matched_t:
                trk.lost += 1

        # ── Spawn new tracks for unmatched detections ──
        for di, (dxy, dcls, dconf) in enumerate(detections):
            if di not in matched_d:
                xywh = xyxy_to_xywh(dxy)
                self.tracks.append((KalmanBoxTracker(xywh), dcls, dconf, None))

        # ── Prune dead tracks ──
        self.tracks = [(t, c, cf, h) for t, c, cf, h in self.tracks if t.lost <= MAX_LOST]

        # ── Build output and update trails ──
        results: list[tuple[list[float], int, int, float, list[float] | None]] = []
        for trk, cls, conf, hidden in self.tracks:
            if trk.hits < MIN_HITS and trk.lost > 0:
                continue

            state    = trk.get_state()               # [cx, cy, w, h]
            box_xyxy = xywh_to_xyxy(state)
            cx, cy   = int(state[0]), int(state[1])

            trail = self.trails[trk.id]
            trail.append((cx, cy))
            if len(trail) > TRAIL_LEN:
                del trail[:-TRAIL_LEN]

            lstm_pred = self._lstm_predict(trk.history, frame_w, frame_h)
            results.append((box_xyxy, trk.id, cls, conf, lstm_pred))

        # ── Prune trails for deleted tracks ──
        live_ids = {trk.id for trk, _, _cf, __ in self.tracks}
        for tid in [k for k in self.trails if k not in live_ids]:
            del self.trails[tid]

        return results

    # ── private ───────────────────────────────────────────────────────────────
    def _lstm_step(self, xywh, fw, fh, hidden):
        """Feed one measurement into the LSTM to advance its hidden state."""
        norm = [[xywh[0] / fw, xywh[1] / fh, xywh[2] / fw, xywh[3] / fh]]
        x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, hidden = self.lstm(x, hidden)
        return hidden

    def _lstm_predict(self, history, fw, fh) -> list[float] | None:
        """Predict next box from the last SEQ_LEN measurements; None if too few."""
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


# ── STL / RTM requirement checker ────────────────────────────────────────────
class STLChecker:
    """
    Evaluates the four STL-formalised RTM requirements for a single track
    per frame. Each individual method returns True when the STL *antecedent*
    is satisfied (meaning the consequent class override should be applied).

    REQ-01  □[0,∞) [(40≤f≤100) ∧ (c=red)  → class=unknown]
            Subsystems: YOLO (colour crop) + LSTM (flash-rate proxy)

    REQ-02  □[0,∞) [(AR≈const) ∧ (v≈0)    → class=unknown]
            Subsystems: YOLO (bbox dims)   + Kalman (kinematics)

    REQ-03  □[0,∞) [(light=off)            → class=bird]
            Subsystem : YOLO (confidence thresholding)

    REQ-04  □[0,∞) [(Δshape>ε)             → class=bird]
            Subsystems: Kalman (association) + LSTM (shape deformation)
    """

    # ── Individual antecedent checkers ───────────────────────────────────────

    @staticmethod
    def req01_flash_and_red(
        trk: KalmanBoxTracker,
        frame: np.ndarray,
        fw: int,
        fh: int,
        fps: float,
    ) -> bool:
        """REQ-01: bbox-width oscillation ∈ [40,100] Hz AND dominant colour is red."""
        history = trk.history
        win = REQ01_BRIGHTNESS_WIN
        if len(history) < win or fps <= 0:
            return False
        recent = history[-win:]
        ws    = np.array([b[2] for b in recent], dtype=np.float32)
        signs = np.sign(ws - ws.mean())
        crossings = int(np.sum(np.diff(signs) != 0))
        flash_hz  = crossings / (2.0 * max(win / fps, 1e-6))
        if not (REQ01_FLASH_HZ_LO <= flash_hz <= REQ01_FLASH_HZ_HI):
            return False
        # Colour check — crop the current frame to the track bounding box.
        cx, cy, w, h = history[-1]
        x1 = max(0, int(cx - w / 2));  y1 = max(0, int(cy - h / 2))
        x2 = min(fw, int(cx + w / 2)); y2 = min(fh, int(cy + h / 2))
        if x2 <= x1 or y2 <= y1:
            return False
        crop = frame[y1:y2, x1:x2]          # BGR
        b_m  = float(crop[:, :, 0].mean())
        g_m  = float(crop[:, :, 1].mean())
        r_m  = float(crop[:, :, 2].mean())
        peak = max(r_m, g_m, b_m) + 1e-6
        return (r_m / peak) >= REQ01_RED_RATIO and r_m > g_m and r_m > b_m

    @staticmethod
    def req02_hover(trk: KalmanBoxTracker) -> bool:
        """REQ-02: aspect-ratio variance ≈ 0 AND Kalman velocity magnitude ≈ 0."""
        history = trk.history
        if len(history) < 4:
            return False
        recent = history[-8:]
        ars    = np.array([b[2] / (b[3] + 1e-6) for b in recent], dtype=np.float32)
        ar_var = float(ars.var())
        vx     = float(trk.x[4, 0])
        vy     = float(trk.x[5, 0])
        vel    = float(np.hypot(vx, vy))
        return ar_var <= REQ02_AR_VAR_MAX and vel <= REQ02_VEL_MAX

    @staticmethod
    def req03_rtamt_classify(
        trk: KalmanBoxTracker,
        frame: np.ndarray,
        fw: int,
        fh: int,
        fps: float,
        frame_idx: int,
    ) -> tuple[bool, int | None]:
        """
        REQ-03: Classifies a light signal as a Drone or Manned Aircraft using RTAMT STL.
        Drone maps to drone (1), MannedAircraft maps to unknown (2).
        """
        # Determine light state in the current frame (1.0 for ON, 0.0 for OFF)
        cx, cy, w, h = trk.history[-1]
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(fw, int(cx + w / 2))
        y2 = min(fh, int(cy + h / 2))
        
        light_state = 0.0
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                max_val = np.max(gray) if gray.size > 0 else 0.0
                # Light is ON if max pixel intensity exceeds 200
                light_state = 1.0 if max_val > 200.0 else 0.0

        timestamp = frame_idx / fps

        # Append to track history
        if not hasattr(trk, "light_history"):
            trk.light_history = []
        if not hasattr(trk, "timestamp_history"):
            trk.timestamp_history = []

        trk.light_history.append(light_state)
        trk.timestamp_history.append(timestamp)

        # Keep history bounded to avoid memory build-up (last 100 frames is plenty)
        if len(trk.light_history) > 100:
            trk.light_history = trk.light_history[-100:]
            trk.timestamp_history = trk.timestamp_history[-100:]

        # We need a minimum history length to evaluate the temporal formulas (e.g. 5 frames)
        if len(trk.light_history) < 5:
            return False, None

        try:
            epsilon = 1.0 / fps
            tau_A = 1.2
            tau_D = 0.8

            # Falling Edge: Light was ON, and immediately drops to OFF
            falling_edge = f"(x == 1.0 and eventually[0:{epsilon:.4f}] (x == 0.0))"

            time_list = list(trk.timestamp_history)
            x_list    = list(trk.light_history)

            # Manned Aircraft: Stays OFF for at least tau_A seconds after falling edge
            spec_ma = rtamt.StlDiscreteTimeSpecification()
            spec_ma.declare_var('x', 'float')
            spec_ma.spec = f"historically({falling_edge} -> always[{epsilon:.4f}:{tau_A:.4f}] (x == 0.0))"
            spec_ma.parse()
            res_ma = spec_ma.evaluate({'time': time_list, 'x': x_list})
            final_aircraft_robustness = res_ma[-1][1] if res_ma else float('-inf')

            # Drone: Must flash ON again within tau_D seconds after falling edge
            spec_dr = rtamt.StlDiscreteTimeSpecification()
            spec_dr.declare_var('x', 'float')
            spec_dr.spec = f"historically({falling_edge} -> eventually[{epsilon:.4f}:{tau_D:.4f}] (x == 1.0))"
            spec_dr.parse()
            res_dr = spec_dr.evaluate({'time': time_list, 'x': x_list})
            final_drone_robustness = res_dr[-1][1] if res_dr else float('-inf')
            
            # 7. Classification Logic
            if final_drone_robustness > 0 and final_aircraft_robustness <= 0:
                # Classify as DRONE (class index 1)
                return True, 1
            elif final_aircraft_robustness > 0 and final_drone_robustness <= 0:
                # Classify as MANNED AIRCRAFT which is mapped to unknown (class index 2)
                return True, 2
            else:
                return False, None
        except Exception as e:
            # Fallback on any RTAMT evaluation error
            return False, None

    @staticmethod
    def req04_shape_deform(trk: KalmanBoxTracker, fw: int, fh: int) -> bool:
        """REQ-04: normalised (w,h) deformation over a window exceeds ε → bird."""
        history = trk.history
        win = REQ04_DEFORM_WIN
        if len(history) < win:
            return False
        recent = history[-win:]
        ws     = np.array([b[2] / fw for b in recent], dtype=np.float32)
        hs     = np.array([b[3] / fh for b in recent], dtype=np.float32)
        deform = float(ws.std() + hs.std())
        return deform > REQ04_SHAPE_EPS

    @staticmethod
    def req05_kinematic_classify(trk: KalmanBoxTracker) -> tuple[bool, int | None]:
        """
        REQ-05: Kinematic STL classifier using Kalman-derived velocity.

        Extracts speed (v) and vertical velocity (vz) from the Kalman state,
        accumulates per-track history, then evaluates four *separate* RTAMT
        specs (one per flight profile).  The class with the highest positive
        robustness wins.

        Bird → 0,  Drone → 1,  Helicopter / Airplane → 2 (unknown).
        """
        # ── Extract current kinematics from Kalman state ──
        vx = float(trk.x[4, 0])   # px / frame
        vy = float(trk.x[5, 0])   # px / frame
        v  = float(np.hypot(vx, vy))
        vz = abs(vy)

        # ── Accumulate per-track history ──
        if not hasattr(trk, "kin_v_history"):
            trk.kin_v_history  = []
            trk.kin_vz_history = []
        trk.kin_v_history.append(v)
        trk.kin_vz_history.append(vz)

        # Bound memory
        if len(trk.kin_v_history) > 200:
            trk.kin_v_history  = trk.kin_v_history[-200:]
            trk.kin_vz_history = trk.kin_vz_history[-200:]

        if len(trk.kin_v_history) < REQ05_MIN_HISTORY:
            return False, None

        try:
            n = len(trk.kin_v_history)
            time_list = list(range(n))
            v_list    = list(trk.kin_v_history)
            vz_list   = list(trk.kin_vz_history)

            is_hov = f"(v <= {REQ05_HOVER_VEL} and vz <= {REQ05_HOVER_VZ})"
            formulas = {
                "Bird":       f"always(v < {REQ05_BIRD_MAX_VEL}) and "
                              f"not(eventually[0:{REQ05_HOVER_WINDOW}] always[0:{REQ05_HOVER_DURATION}] {is_hov})",
                "Drone":      f"(eventually[0:{REQ05_HOVER_WINDOW}] always[0:{REQ05_HOVER_DURATION}] {is_hov}) and always(v < {REQ05_FAST_VEL})",
                "Helicopter": f"(eventually[0:{REQ05_HOVER_WINDOW}] (v > {REQ05_FAST_VEL})) and "
                              f"(eventually[0:{REQ05_HOVER_WINDOW}] always[0:{REQ05_HOVER_DURATION}] {is_hov})",
                "Airplane":   f"always(v > {REQ05_CRUISE_VEL}) and always(not {is_hov})",
            }

            robustness: dict[str, float] = {}
            for name, formula in formulas.items():
                spec = rtamt.StlDiscreteTimeSpecification()
                spec.declare_var('v', 'float')
                spec.declare_var('vz', 'float')
                spec.spec = formula
                spec.parse()
                trace = spec.evaluate({'time': time_list, 'v': v_list, 'vz': vz_list})
                robustness[name] = trace[0][1] if trace else float('-inf')

            best = max(robustness, key=robustness.get)
            if robustness[best] <= 0:
                return False, None

            class_map = {"Bird": 0, "Drone": 1, "Helicopter": 2, "Airplane": 2}
            return True, class_map[best]

        except Exception:
            return False, None

    # ── Combined check (priority: REQ-03 > REQ-01 > REQ-02 > REQ-04 > REQ-05) ─

    @staticmethod
    def check(
        trk: KalmanBoxTracker,
        conf: float,
        frame: np.ndarray,
        fw: int,
        fh: int,
        fps: float,
        frame_idx: int,
    ) -> tuple:
        """
        Evaluate all five requirements *independently* (for per-req counting),
        then apply the priority-ordered class override.

        Returns
        -------
        fires    : dict[str, bool]  — which antecedents were satisfied this frame
        override : int | None       — class override (0=bird, 1=drone, 2=unknown) or None
        fired_id : str | None       — req that produced the override, or None
        """
        r01 = STLChecker.req01_flash_and_red(trk, frame, fw, fh, fps)
        r02 = STLChecker.req02_hover(trk)
        r03, override_r03 = STLChecker.req03_rtamt_classify(trk, frame, fw, fh, fps, frame_idx)
        r04 = STLChecker.req04_shape_deform(trk, fw, fh)
        r05, override_r05 = STLChecker.req05_kinematic_classify(trk)
        fires = {
            "REQ-01": (r01, 2 if r01 else None),
            "REQ-02": (r02, 2 if r02 else None),
            "REQ-03": (r03, override_r03 if r03 else None),
            "REQ-04": (r04, 0 if r04 else None),
            "REQ-05": (r05, override_r05 if r05 else None),
        }
        if r03 and override_r03 is not None:
            return fires, override_r03, "REQ-03"
        if r01:
            return fires, 2, "REQ-01"
        if r02:
            return fires, 2, "REQ-02"
        if r04:
            return fires, 0, "REQ-04"
        if r05 and override_r05 is not None:
            return fires, override_r05, "REQ-05"
        return fires, None, None


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_tracks(
    frame: np.ndarray,
    track_results: list,
    trails: dict,
):
    for box_xyxy, track_id, cls_int, conf, lstm_pred in track_results:
        label = CLASS_NAMES[cls_int] if cls_int < len(CLASS_NAMES) else "unknown"
        color = COLORS[label]

        # Bounding box.
        x1, y1, x2, y2 = (int(v) for v in box_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label with confidence.
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_bg_y1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, text_bg_y1), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame, text, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

        # Fading trail.
        trail = trails.get(track_id, [])
        n = len(trail)
        for i in range(1, n):
            alpha = i / n
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, trail[i - 1], trail[i], c, 2)

        # LSTM predicted next centre – hollow circle.
        if lstm_pred is not None:
            lx1, ly1, lx2, ly2 = lstm_pred
            lcx = int((lx1 + lx2) / 2)
            lcy = int((ly1 + ly2) / 2)
            cv2.circle(frame, (lcx, lcy), 6, color, 2)


# ── GT label helper ──────────────────────────────────────────────────────────
def load_gt_label_for_frame(img_path: Path, frame_w: int, frame_h: int):
    """Load YOLO .txt GT for a specific frame if present.
    Returns (class_id, [x_min, y_min, width_px, height_px]) or None."""
    txt_path = LABELS_DIR / img_path.with_suffix(".txt").name
    if not txt_path.exists():
        return None
    try:
        with open(txt_path) as f:
            lines = f.readlines()
        if not lines:
            return None
        # Just take the first object (assume 1 drone per frame)
        parts = lines[0].strip().split()
        if len(parts) >= 5:
            # YOLO format: class cx cy w h (normalized)
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            rx = (cx - w / 2) * frame_w
            ry = (cy - h / 2) * frame_h
            rw = w * frame_w
            rh = h * frame_h
            return cls_id, [rx, ry, rw, rh]
    except Exception:
        pass
    return None


# ── Questionary helpers (same as inference_video.py) ─────────────────────────
def choose_run_folder() -> str:
    weights_dir = PROJECT_ROOT / "runs" / "detect" / "weights"
    available_pts = sorted(
        [p.name for p in weights_dir.glob("*.pt")]
    )

    if not available_pts:
        raise ValueError(f"No .pt files found in {weights_dir}")

    if len(sys.argv) > 1:
        run_name = sys.argv[1]
        # Accept a bare filename (e.g. "yolo8n.pt") or a full path
        if run_name.endswith(".pt"):
            candidate = Path(run_name)
            if not candidate.is_absolute():
                candidate = weights_dir / run_name
            if not candidate.exists():
                raise FileNotFoundError(f"Model file not found: {candidate}")
            return str(candidate)
        raise ValueError(
            f"Expected a .pt filename. Available: {', '.join(available_pts)}"
        )

    chosen = questionary.select(
        "Choose a trained YOLO model (.pt) from runs/detect/weights:",
        choices=available_pts,
    ).ask()
    if not chosen:
        raise ValueError("No model selected")
    return str(weights_dir / chosen)


def choose_sequence() -> list[str]:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    # Find all unique video prefixes
    prefixes = set()
    for p in IMAGES_DIR.glob("*.jpg"):
        m = re.match(r'^(.*?)-\d+_jpg\.rf\..*\.jpg$', p.name)
        if m:
            prefixes.add(m.group(1))

    prefixes = sorted(list(prefixes))
    if not prefixes:
        raise ValueError(f"No valid sequence images found in {IMAGES_DIR}")

    if len(sys.argv) > 2:
        seq_name = sys.argv[2]
        if seq_name == "all":
            return prefixes
        if seq_name not in prefixes:
            raise FileNotFoundError(f"Sequence prefix not found: {seq_name}")
        return [seq_name]

    if len(prefixes) == 1:
        return prefixes

    choices = ["All Sequences"] + prefixes
    seq_name = questionary.select(
        "Choose a sequence prefix from dataset 2/test/images:",
        choices=choices,
    ).ask()
    
    if not seq_name:
        raise ValueError("No sequence selected")
        
    if seq_name == "All Sequences":
        return prefixes
    return [seq_name]


# ── Main ──────────────────────────────────────────────────────────────────────
def process_sequence(seq_prefix: str, run_name: str, loaded_models: list, lstm_device: torch.device, show_video: bool = True):
    output_path = PROJECT_ROOT / "runs" / "detect" / "inference" / f"{seq_prefix}_{run_name}_tracking.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Re-initialize tracker for each sequence
    tracker = MultiObjectTracker(lstm_device=lstm_device)

    # Glob all sequence images
    seq_images = []
    for p in IMAGES_DIR.glob("*.jpg"):
        m = re.match(r'^(.*?)-(\d+)_jpg\.rf\..*\.jpg$', p.name)
        if m and m.group(1) == seq_prefix:
            seq_images.append((int(m.group(2)), p))
    seq_images.sort(key=lambda x: x[0])
    
    if not seq_images:
        print(f"No images found for sequence {seq_prefix}")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(seq_images[0][1]))
    if first_frame is None:
        print(f"Could not read first frame: {seq_images[0][1]}")
        return

    height, width = first_frame.shape[:2]
    fps = 30.0
    total_frames = len(seq_images)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height)
    )
    if not writer.isOpened():
        print(f"Could not create output video: {output_path}")
        return

    print(f"{'Input sequence:':<22}{seq_prefix} ({total_frames} frames)")
    print(f"{'Saving output to:':<22}{output_path}")
    print()

    frame_ious_base:     list[float] = []
    frame_dists_base:    list[float] = []
    covered_frames_base: int = 0
    frame_ious_final:     list[float] = []
    frame_dists_final:    list[float] = []
    covered_frames_final: int = 0
    exist1_frames:  int = 0

    # STL / RTM per-requirement counters
    rtm_evals     = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
    rtm_fires     = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
    rtm_satisfactions = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
    rtm_violations    = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
    rtm_overrides = 0
    base_false_alarms = 0
    mitigated_strict = 0
    mitigated_safety = 0
    cls_matched_frames = 0
    cls_base_correct = 0
    cls_final_correct = 0
    cls_safety_correct = 0
    fa_frames = 0  # frames that are false alarms (wrong class or spurious detection)

    processed  = 0
    start_time = time.time()
    
    has_gt = False  # Track if any GT labels were found

    for frame_idx, img_path in seq_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        gt_res = load_gt_label_for_frame(img_path, width, height)
        if gt_res is not None:
            has_gt = True

        # WBF: fuse detections from all ensemble models.
        fused      = run_wbf_on_frame(loaded_models, frame)
        detections = [(d["box"], d["class_id"], d["confidence"]) for d in fused]

        track_results = tracker.update(detections, width, height)

        # ── STL / RTM override pass ────────────────────────────────────────────
        # Map track_id → KalmanBoxTracker for STLChecker access.
        track_obj_map: dict[int, KalmanBoxTracker] = {
            trk.id: trk for trk, _, _cf, __ in tracker.tracks
        }
        base_cls_map = {}
        fires_map: dict[int, dict] = {}
        overridden: list = []
        for box_xyxy, track_id, cls_int, conf, lstm_pred in track_results:
            base_cls_map[track_id] = cls_int
            trk_obj = track_obj_map.get(track_id)
            if trk_obj is not None:
                fires, override, fired_id = STLChecker.check(
                    trk_obj, conf, frame, width, height, fps, frame_idx
                )
                fires_map[track_id] = fires
                # Count antecedent evaluations and fires per requirement.
                for rid, (fired, _req_override) in fires.items():
                    rtm_evals[rid] += 1
                    if fired:
                        rtm_fires[rid] += 1
                if override is not None:
                    rtm_overrides += 1
                    cls_int = override
            overridden.append((box_xyxy, track_id, cls_int, conf, lstm_pred))
        track_results = overridden
        # ─────────────────────────────────────────────────────────────────

        draw_tracks(frame, track_results, tracker.trails)

        # ── FA/h tracking ──────────────────────────────────────────────────────
        if gt_res is None:
            # GT absent (no label for this frame): any detection is a false alarm.
            if track_results:
                fa_frames += 1
        else:
            fa_gt_cls, fa_gt_rect = gt_res
            fa_rx, fa_ry, fa_rw, fa_rh = fa_gt_rect
            fa_gt_xyxy = [float(fa_rx), float(fa_ry), float(fa_rx + fa_rw), float(fa_ry + fa_rh)]
            is_fa = False
            for fa_box, _ftid, fa_cls, _fconf, _flstm in track_results:
                fa_iou = compute_iou(fa_box, fa_gt_xyxy)
                if fa_cls != fa_gt_cls or fa_iou < FA_IOU_THRESH:
                    is_fa = True
                    break
            fa_frames += int(is_fa)
        # ──────────────────────────────────────────────────────────────────────

        # ── GT overlay + per-frame evaluation ──────────────────────────────
        if gt_res is not None:
            gt_cls, gt_rect = gt_res
            rx, ry, rw, rh = gt_rect
            gx1, gy1 = int(rx), int(ry)
            gx2, gy2 = int(rx + rw), int(ry + rh)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 255, 255), 2)
            cv2.putText(
                frame, f"GT: {CLASS_NAMES[gt_cls] if gt_cls < len(CLASS_NAMES) else 'unknown'}", 
                (gx1, max(12, gy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            )

            gt_xyxy = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
            gt_cx   = rx + rw / 2.0
            gt_cy   = ry + rh / 2.0
            exist1_frames += 1

            best_iou_base  = 0.0
            best_dist_base = float("inf")
            best_iou_final = 0.0
            best_dist_final = float("inf")
            
            best_iou_agnostic = 0.0
            matched_track_agnostic = None

            for box_xyxy, tid, cls_final, _conf, lstm_pred in track_results:
                iou_val = compute_iou(box_xyxy, gt_xyxy)
                tcx = (box_xyxy[0] + box_xyxy[2]) / 2.0
                tcy = (box_xyxy[1] + box_xyxy[3]) / 2.0
                dist_val = float(np.hypot(tcx - gt_cx, tcy - gt_cy))
                
                if iou_val > best_iou_agnostic:
                    best_iou_agnostic = iou_val
                    matched_track_agnostic = (box_xyxy, tid, cls_final)

                cls_base = base_cls_map.get(tid, cls_final)
                
                if cls_base == gt_cls:
                    if iou_val > best_iou_base:
                        best_iou_base  = iou_val
                        best_dist_base = dist_val
                
                if cls_final == gt_cls:
                    if iou_val > best_iou_final:
                        best_iou_final  = iou_val
                        best_dist_final = dist_val

            frame_ious_base.append(best_iou_base)
            frame_dists_base.append(best_dist_base)
            if best_iou_base > 0:
                covered_frames_base += 1

            frame_ious_final.append(best_iou_final)
            frame_dists_final.append(best_dist_final)
            if best_iou_final > 0:
                covered_frames_final += 1

            # Compute False Alarms and Mitigations
            if best_iou_agnostic >= 0.5 and matched_track_agnostic is not None:
                _, tid, cls_final = matched_track_agnostic
                cls_base = base_cls_map.get(tid, cls_final)
                cls_matched_frames += 1
                
                if cls_base == gt_cls:
                    cls_base_correct += 1
                if cls_final == gt_cls:
                    cls_final_correct += 1
                if cls_final == gt_cls or cls_final == 2:
                    cls_safety_correct += 1

                if cls_base != gt_cls:
                    base_false_alarms += 1
                    if cls_final == gt_cls:
                        mitigated_strict += 1
                    if cls_final == gt_cls or cls_final == 2:
                        mitigated_safety += 1

                # Per-REQ satisfaction / violation tracking
                if tid in fires_map:
                    for rid, (rfired, req_override) in fires_map[tid].items():
                        if rfired and req_override is not None:
                            if req_override == gt_cls:
                                rtm_satisfactions[rid] += 1
                            else:
                                rtm_violations[rid] += 1

        writer.write(frame)

        if show_video:
            win_name = f"Tracking Visualization - {seq_prefix} (Space: Pause, Q: Quit)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            
            # Auto-resize window so large frames don't overflow the screen
            if width > 1280 or height > 720:
                scale = min(1280 / width, 720 / height)
                cv2.resizeWindow(win_name, int(width * scale), int(height * scale))
            
            cv2.imshow(win_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nVisualization stopped by user.")
                break
            elif key == ord(' '):  # Spacebar to pause
                print("\nPaused. Press Space to resume...")
                paused = True
                while paused:
                    key2 = cv2.waitKey(30) & 0xFF
                    if key2 == ord(' '):
                        print("Resumed.")
                        paused = False
                    elif key2 == ord('q') or key2 == 27:
                        print("\nVisualization stopped by user.")
                        paused = False
                        break
                if key2 == ord('q') or key2 == 27:
                    break

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

    writer.release()
    if show_video:
        cv2.destroyAllWindows()

    print()
    if total_frames > 0:
        print(f"Done. Processed {processed}/{total_frames} frames.")
    else:
        print(f"Done. Processed {processed} frames.")

    # Compute FA/h unconditionally so return dict always has these fields.
    duration_h  = (processed / fps / 3600.0) if fps > 0 else 0.0
    fa_per_hour = fa_frames / duration_h if duration_h > 0 else 0.0

    # ── Evaluation summary ──────────────────────────────────────────────────
    if has_gt and exist1_frames > 0:
        def calc_metrics(ious, dists, cov_frames):
            iou_arr  = np.array(ious, dtype=np.float32)
            dist_arr = np.array(dists, dtype=np.float32)
            thr_iou  = np.linspace(0.0, 1.0, 101)
            success  = np.array([(iou_arr >= t).mean() for t in thr_iou], dtype=np.float32)
            auc      = float(np.trapezoid(success, thr_iou))
            sr50     = float((iou_arr >= 0.5).mean())
            thr_dist  = np.arange(0, 51, dtype=np.float32)
            precision = np.array([(dist_arr <= t).mean() for t in thr_dist], dtype=np.float32)
            prec20    = float((dist_arr <= 20.0).mean())
            coverage  = cov_frames / exist1_frames
            mean_iou  = float(iou_arr.mean())
            return mean_iou, sr50, auc, prec20, coverage, success, precision, thr_iou, thr_dist

        (mean_iou_base, sr50_base, auc_base, prec20_base, coverage_base, 
         success_base, precision_base, thr_iou, thr_dist) = calc_metrics(frame_ious_base, frame_dists_base, covered_frames_base)
         
        (mean_iou_final, sr50_final, auc_final, prec20_final, coverage_final, 
         success_final, precision_final, _, _) = calc_metrics(frame_ious_final, frame_dists_final, covered_frames_final)

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  Evaluation  ({exist1_frames} present frames / {processed} total)")
        print(sep)
        print("  Tracking Metrics (Prior to STL):")
        print(f"  {'Mean IoU:':<32}{mean_iou_base * 100:.2f}%")
        print(f"  {'Success Rate  @IoU≥0.5:':<32}{sr50_base * 100:.2f}%")
        print(f"  {'AUC  (success curve 0→1):':<32}{auc_base * 100:.2f}%")
        print(f"  {'Precision  @20 px:':<32}{prec20_base * 100:.2f}%")
        print(f"  {'Coverage:':<32}{coverage_base * 100:.2f}%  ({covered_frames_base}/{exist1_frames})")
        print(sep)
        print("  Tracking Metrics (After STL Mitigation):")
        print(f"  {'Mean IoU:':<32}{mean_iou_final * 100:.2f}%")
        print(f"  {'Success Rate  @IoU≥0.5:':<32}{sr50_final * 100:.2f}%")
        print(f"  {'AUC  (success curve 0→1):':<32}{auc_final * 100:.2f}%")
        print(f"  {'Precision  @20 px:':<32}{prec20_final * 100:.2f}%")
        print(f"  {'Coverage:':<32}{coverage_final * 100:.2f}%  ({covered_frames_final}/{exist1_frames})")
        print(sep)

        print(f"  {'FA frames:':<32}{fa_frames}  /  {processed} total")
        print(f"  {'Duration:':<32}{duration_h * 60:.1f} min")
        print(f"  {'FA/h:':<32}{fa_per_hour:.2f}")
        print(sep)

        base_cls_acc = (cls_base_correct / cls_matched_frames * 100.0) if cls_matched_frames > 0 else 0.0
        final_cls_acc = (cls_final_correct / cls_matched_frames * 100.0) if cls_matched_frames > 0 else 0.0
        safety_cls_acc = (cls_safety_correct / cls_matched_frames * 100.0) if cls_matched_frames > 0 else 0.0
        
        strict_mit_rate = (mitigated_strict / base_false_alarms * 100.0) if base_false_alarms > 0 else 0.0
        safety_mit_rate = (mitigated_safety / base_false_alarms * 100.0) if base_false_alarms > 0 else 0.0

        print(f"  {'Matched Track Frames:':<32}{cls_matched_frames}")
        print(sep)
        print("  Classification Evaluation (Prior to STL):")
        print(f"  {'Base Cls Accuracy:':<32}{base_cls_acc:.2f}%")
        print(f"  {'Base False Alarms:':<32}{base_false_alarms}")
        print(sep)
        print("  Classification Evaluation (After STL Mitigation):")
        print(f"  {'Final Cls Accuracy:':<32}{final_cls_acc:.2f}%")
        print(f"  {'Safety-Aware Cls Accuracy:':<32}{safety_cls_acc:.2f}%")
        print(f"  {'Strict Mitigated:':<32}{mitigated_strict}")
        print(f"  {'Safety Mitigated:':<32}{mitigated_safety}")
        print(f"  {'Strict Mitigation Rate:':<32}{strict_mit_rate:.2f}%")
        print(f"  {'Safety Mitigation Rate:':<32}{safety_mit_rate:.2f}%")
        print(sep)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(thr_iou, success_base, linewidth=2, label="Base", linestyle="--")
        ax1.plot(thr_iou, success_final, linewidth=2, label="Final")
        ax1.set_xlabel("IoU threshold")
        ax1.set_ylabel("Success rate")
        ax1.set_title(f"AUC: {auc_base:.3f} -> {auc_final:.3f}")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.legend(); ax1.grid(True, alpha=0.4)

        ax2.plot(thr_dist, precision_base, linewidth=2, label="Base", linestyle="--")
        ax2.plot(thr_dist, precision_final, linewidth=2, label="Final")
        ax2.axvline(20, color="gray", linestyle=":", linewidth=1, label="20 px")
        ax2.set_xlabel("Centre distance threshold (px)")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"Prec @20px: {prec20_base:.3f} -> {prec20_final:.3f}")
        ax2.set_xlim(0, 50); ax2.set_ylim(0, 1)
        ax2.legend(); ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        plot_path = output_path.parent / f"{seq_prefix}_{run_name}_eval.png"
        fig.savefig(str(plot_path), dpi=120)
        plt.close(fig)
        print(f"\n  Eval plot → {plot_path}")

        csv_path = output_path.parent / f"{seq_prefix}_{run_name}_eval.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value_base", "value_final", "note"])
            writer.writerow(["mean_iou", mean_iou_base, mean_iou_final, ""])
            writer.writerow(["success_rate_iou_ge_0_5", sr50_base, sr50_final, "IoU >= 0.5"])
            writer.writerow(["auc_success_curve_0_to_1", auc_base, auc_final, ""])
            writer.writerow(["precision_at_20px", prec20_base, prec20_final, "center distance <= 20"])
            writer.writerow(["coverage", coverage_base, coverage_final, f"covered/present frames"])
            writer.writerow(["present_frames", exist1_frames, exist1_frames, ""])
            writer.writerow(["processed_frames", processed, processed, ""])
            writer.writerow(["cls_matched_frames", cls_matched_frames, cls_matched_frames, ""])
            writer.writerow(["base_cls_accuracy", base_cls_acc / 100.0, base_cls_acc, ""])
            writer.writerow(["final_cls_accuracy", final_cls_acc / 100.0, final_cls_acc, ""])
            writer.writerow(["safety_aware_cls_accuracy", safety_cls_acc / 100.0, safety_cls_acc, ""])
            writer.writerow(["base_false_alarms", base_false_alarms, "", ""])
            writer.writerow(["mitigated_strict", mitigated_strict, "", ""])
            writer.writerow(["mitigated_safety", mitigated_safety, "", ""])
            writer.writerow(["strict_mitigation_rate", strict_mit_rate / 100.0, strict_mit_rate, ""])
            writer.writerow(["safety_mitigation_rate", safety_mit_rate / 100.0, safety_mit_rate, ""])
            writer.writerow(["fa_frames", fa_frames, "", ""])
            writer.writerow(["duration_h", duration_h, duration_h * 60, "hours / minutes"])
            writer.writerow(["fa_per_hour", fa_per_hour, "", "false alarms per hour"])
        print(f"  Eval csv  → {csv_path}")

    # ── STL / RTM results summary ────────────────────────────────────────────────────
    REQ_LABELS = {
        "REQ-01": "Flash[1-14Hz] ∧ Red    → unknown",
        "REQ-02": "AR≈const ∧ hover       → unknown",
        "REQ-03": "RTAMT Flash Pattern    → drone/unk",
        "REQ-04": "Δshape > ε             → bird   ",
        "REQ-05": "Kinematic STL classify → b/d/u  ",
    }
    sep2 = "─" * 85
    print(f"\n{sep2}")
    print(f"  STL / RTM Requirements — Satisfactions & Violations")
    print(sep2)
    print(f"  {'Req':<8}  {'Description':<35}  {'Fires':>6}  {'Evals':>6}  {'Rate':>7}  {'Sat':>5}  {'Viol':>5}")
    print(f"  {'-'*8:<8}  {'-'*35:<35}  {'-'*6:>6}  {'-'*6:>6}  {'-'*7:>7}  {'-'*5:>5}  {'-'*5:>5}")
    for rid in ("REQ-01", "REQ-02", "REQ-03", "REQ-04", "REQ-05"):
        ev  = rtm_evals[rid]
        fi  = rtm_fires[rid]
        sat = rtm_satisfactions[rid]
        vio = rtm_violations[rid]
        pct = (fi / ev * 100.0) if ev > 0 else 0.0
        print(f"  {rid:<8}  {REQ_LABELS[rid]:<35}  {fi:>6}  {ev:>6}  {pct:>6.2f}%  {sat:>5}  {vio:>5}")
    total_evals = sum(rtm_evals.values())
    total_fires = sum(rtm_fires.values())
    total_sat   = sum(rtm_satisfactions.values())
    total_vio   = sum(rtm_violations.values())
    overall_pct = (total_fires / total_evals * 100.0) if total_evals > 0 else 0.0
    print(sep2)
    print(f"  {'Total overrides applied:':<48}{rtm_overrides:>6}")
    print(f"  {'Total antecedent fires (all reqs):':<48}{total_fires:>6}  /  {total_evals} evals  ({overall_pct:.2f}%)")
    print(f"  {'Total satisfactions (fire matched GT):':<48}{total_sat:>6}")
    print(f"  {'Total violations (fire ≠ GT):':<48}{total_vio:>6}")
    print(sep2)

    # Append RTM rows to the eval CSV if it was written.
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w2 = csv.writer(f)
            w2.writerow([])
            w2.writerow(["# STL/RTM requirement results"])
            w2.writerow(["req_id", "description", "fires", "evaluations", "fire_rate_percent", "satisfactions", "violations"])
            for rid in ("REQ-01", "REQ-02", "REQ-03", "REQ-04", "REQ-05"):
                ev  = rtm_evals[rid]
                fi  = rtm_fires[rid]
                sat = rtm_satisfactions[rid]
                vio = rtm_violations[rid]
                pct = (fi / ev * 100.0) if ev > 0 else 0.0
                w2.writerow([rid, REQ_LABELS[rid].strip(), fi, ev, f"{pct:.4f}", sat, vio])
            w2.writerow(["RTM_total_overrides", "", rtm_overrides, total_evals, f"{overall_pct:.4f}", total_sat, total_vio])
        print(f"  RTM rows appended → {csv_path}")
    except NameError:
        # csv_path is only defined when GT labels were present; skip silently.
        rtm_csv = output_path.parent / f"{seq_prefix}_{run_name}_rtm.csv"
        with open(rtm_csv, "w", newline="", encoding="utf-8") as f:
            w2 = csv.writer(f)
            w2.writerow(["req_id", "description", "fires", "evaluations", "fire_rate_percent", "satisfactions", "violations"])
            for rid in ("REQ-01", "REQ-02", "REQ-03", "REQ-04", "REQ-05"):
                ev  = rtm_evals[rid]
                fi  = rtm_fires[rid]
                sat = rtm_satisfactions[rid]
                vio = rtm_violations[rid]
                pct = (fi / ev * 100.0) if ev > 0 else 0.0
                w2.writerow([rid, REQ_LABELS[rid].strip(), fi, ev, f"{pct:.4f}", sat, vio])
            w2.writerow(["RTM_total_overrides", "", rtm_overrides, total_evals, f"{overall_pct:.4f}", total_sat, total_vio])
        print(f"  RTM csv  → {rtm_csv}")

    return {
        "processed_frames": processed,
        "exist1_frames": exist1_frames,
        "has_gt": has_gt,
        "mean_iou_base": mean_iou_base if (has_gt and exist1_frames > 0) else None,
        "success_rate_base": sr50_base if (has_gt and exist1_frames > 0) else None,
        "auc_base": auc_base if (has_gt and exist1_frames > 0) else None,
        "precision_20px_base": prec20_base if (has_gt and exist1_frames > 0) else None,
        "coverage_base": coverage_base if (has_gt and exist1_frames > 0) else None,
        "mean_iou_final": mean_iou_final if (has_gt and exist1_frames > 0) else None,
        "success_rate_final": sr50_final if (has_gt and exist1_frames > 0) else None,
        "auc_final": auc_final if (has_gt and exist1_frames > 0) else None,
        "precision_20px_final": prec20_final if (has_gt and exist1_frames > 0) else None,
        "coverage_final": coverage_final if (has_gt and exist1_frames > 0) else None,
        "rtm_evals": rtm_evals,
        "rtm_fires": rtm_fires,
        "rtm_overrides": rtm_overrides,
        "rtm_satisfactions": rtm_satisfactions,
        "rtm_violations": rtm_violations,
        "cls_matched_frames": cls_matched_frames,
        "cls_base_correct": cls_base_correct,
        "cls_final_correct": cls_final_correct,
        "cls_safety_correct": cls_safety_correct,
        "base_false_alarms": base_false_alarms,
        "mitigated_strict": mitigated_strict,
        "mitigated_safety": mitigated_safety,
        "fa_frames": fa_frames,
        "fa_per_hour": fa_per_hour,
        "duration_h": duration_h,
    }



def main():
    seq_prefixes = choose_sequence()

    # Resolve LSTM / torch device.
    if DEVICE:
        lstm_device = torch.device("cuda:0" if DEVICE == "0" else DEVICE)
    else:
        lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load WBF ensemble models.
    print("Loading WBF ensemble models...")
    loaded_models: list[tuple[str, object, dict]] = []
    for model_name, model_path in MODELS:
        if not ENABLED_MODELS.get(model_name, False):
            continue
        if not model_path.exists():
            print(f"  [SKIP] {model_name}: not found at {model_path}")
            continue
        mdl       = YOLO(str(model_path))
        class_map = build_model_class_map(mdl)
        loaded_models.append((model_name, mdl, class_map))
        print(f"  [OK]   {model_name}")

    if not loaded_models:
        raise ValueError(
            "No ensemble models were loaded. Check MODELS paths and ENABLED_MODELS."
        )

    global MIN_MODEL_SUPPORT
    majority = (len(loaded_models) + 1) // 2
    if MIN_MODEL_SUPPORT > len(loaded_models):
        print(
            f"  [NOTE] MIN_MODEL_SUPPORT adjusted from {MIN_MODEL_SUPPORT} "
            f"→ {majority} for {len(loaded_models)} loaded models."
        )
        MIN_MODEL_SUPPORT = majority

    run_name = "wbf_" + "_".join(n for n, _, __ in loaded_models)
    print(f"\nEnsemble run name: {run_name}")
    print()

    # Determine whether to show visualization window.
    show_video = True
    if "--no-show" in sys.argv:
        show_video = False
        sys.argv.remove("--no-show")
    else:
        try:
            show_video = questionary.confirm(
                "Would you like to visualize the tracking in real-time?",
                default=True
            ).ask()
            if show_video is None:
                show_video = False
        except Exception:
            show_video = False

    all_results = []
    for seq_prefix in seq_prefixes:
        res = process_sequence(seq_prefix, run_name, loaded_models, lstm_device, show_video=show_video)
        if res is not None:
            all_results.append(res)

    if len(all_results) > 1:
        # Print grand consolidated mean average report
        print("\n" + "=" * 62)
        print("  GRAND SUMMARY — MEAN AVERAGE OVER ALL SEQUENCES")
        print("=" * 62)
        
        # Filter sequences with GT
        gt_results = [r for r in all_results if r["has_gt"] and r["exist1_frames"] > 0]
        
        if gt_results:
            avg_iou_base = np.mean([r["mean_iou_base"] for r in gt_results])
            avg_sr_base = np.mean([r["success_rate_base"] for r in gt_results])
            avg_auc_base = np.mean([r["auc_base"] for r in gt_results])
            avg_prec_base = np.mean([r["precision_20px_base"] for r in gt_results])
            avg_cov_base = np.mean([r["coverage_base"] for r in gt_results])

            avg_iou_final = np.mean([r["mean_iou_final"] for r in gt_results])
            avg_sr_final = np.mean([r["success_rate_final"] for r in gt_results])
            avg_auc_final = np.mean([r["auc_final"] for r in gt_results])
            avg_prec_final = np.mean([r["precision_20px_final"] for r in gt_results])
            avg_cov_final = np.mean([r["coverage_final"] for r in gt_results])
            
            print(f"  Tracking Evaluation (Prior to STL) (Averaged over {len(gt_results)} GT sequences):")
            print(f"    Mean IoU:                     {avg_iou_base * 100.0:.2f}%")
            print(f"    Success Rate @IoU>=0.5:       {avg_sr_base * 100.0:.2f}%")
            print(f"    AUC (success curve 0->1):     {avg_auc_base * 100.0:.2f}%")
            print(f"    Precision @20 px:              {avg_prec_base * 100.0:.2f}%")
            print(f"    Coverage:                     {avg_cov_base * 100.0:.2f}%")
            print("-" * 62)
            print(f"  Tracking Evaluation (After STL Mitigation) (Averaged over {len(gt_results)} GT sequences):")
            print(f"    Mean IoU:                     {avg_iou_final * 100.0:.2f}%")
            print(f"    Success Rate @IoU>=0.5:       {avg_sr_final * 100.0:.2f}%")
            print(f"    AUC (success curve 0->1):     {avg_auc_final * 100.0:.2f}%")
            print(f"    Precision @20 px:              {avg_prec_final * 100.0:.2f}%")
            print(f"    Coverage:                     {avg_cov_final * 100.0:.2f}%")
            print("-" * 62)

        # Average Classification Metrics over sequences with matched track frames
        gt_cls_results = [r for r in all_results if r["has_gt"] and r["cls_matched_frames"] > 0]
        if gt_cls_results:
            avg_base_acc = np.mean([r["cls_base_correct"] / r["cls_matched_frames"] * 100.0 for r in gt_cls_results])
            avg_final_acc = np.mean([r["cls_final_correct"] / r["cls_matched_frames"] * 100.0 for r in gt_cls_results])
            avg_safety_acc = np.mean([r["cls_safety_correct"] / r["cls_matched_frames"] * 100.0 for r in gt_cls_results])
            
            print(f"  Classification Evaluation (Prior to STL) (Averaged over {len(gt_cls_results)} GT sequences):")
            print(f"    Base Cls Accuracy:            {avg_base_acc:.2f}%")
            print("-" * 62)
            print(f"  Classification Evaluation (After STL Mitigation) (Averaged over {len(gt_cls_results)} GT sequences):")
            print(f"    Final Cls Accuracy:           {avg_final_acc:.2f}%")
            print(f"    Safety-Aware Cls Accuracy:    {avg_safety_acc:.2f}%")
            print("-" * 62)

        # Aggregate STL requirements stats
        avg_rtm_evals = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
        avg_rtm_fires = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
        avg_rtm_sat   = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
        avg_rtm_vio   = {"REQ-01": 0, "REQ-02": 0, "REQ-03": 0, "REQ-04": 0, "REQ-05": 0}
        total_overrides = 0
        total_base_fa = 0
        total_mitigated_strict = 0
        total_mitigated_safety = 0
        
        for r in all_results:
            total_overrides += r["rtm_overrides"]
            total_base_fa += r["base_false_alarms"]
            total_mitigated_strict += r["mitigated_strict"]
            total_mitigated_safety += r["mitigated_safety"]
            for rid in avg_rtm_evals:
                avg_rtm_evals[rid] += r["rtm_evals"][rid]
                avg_rtm_fires[rid] += r["rtm_fires"][rid]
                avg_rtm_sat[rid]   += r["rtm_satisfactions"][rid]
                avg_rtm_vio[rid]   += r["rtm_violations"][rid]
                
        print("  Consolidated STL/RTM Requirements:")
        print(f"    {'Requirement':<10}  {'Evals':>6}  {'Fires':>6}  {'Fire Rate':>10}  {'Sat':>5}  {'Viol':>5}")
        print(f"    {'-'*10:<10}  {'-'*6:>6}  {'-'*6:>6}  {'-'*10:>10}  {'-'*5:>5}  {'-'*5:>5}")
        for rid in sorted(avg_rtm_evals.keys()):
            ev = avg_rtm_evals[rid]
            fi = avg_rtm_fires[rid]
            sat = avg_rtm_sat[rid]
            vio = avg_rtm_vio[rid]
            rate = (fi / ev * 100.0) if ev > 0 else 0.0
            print(f"    {rid:<10}  {ev:>6}  {fi:>6}  {rate:>9.2f}%  {sat:>5}  {vio:>5}")
            
        print("-" * 62)
        print("  Alarms and Mitigation Summary:")
        print(f"    Total overrides applied:             {total_overrides}")
        print(f"    Total base false alarms:             {total_base_fa}")
        print(f"    Total strict mitigated:              {total_mitigated_strict}")
        print(f"    Total safety mitigated:              {total_mitigated_safety}")
        strict_mit_rate = (total_mitigated_strict / total_base_fa * 100.0) if total_base_fa > 0 else 0.0
        safety_mit_rate = (total_mitigated_safety / total_base_fa * 100.0) if total_base_fa > 0 else 0.0
        print(f"    Overall Strict Mitigation Rate:      {strict_mit_rate:.2f}%")
        print(f"    Overall Safety Mitigation Rate:      {safety_mit_rate:.2f}%")

        # Aggregate FA/h across all sequences
        total_fa_frames  = sum(r.get("fa_frames", 0) for r in all_results)
        total_duration_h = sum(r.get("duration_h", 0.0) for r in all_results)
        total_fa_per_h   = total_fa_frames / total_duration_h if total_duration_h > 0 else 0.0
        print(f"    Total FA frames (all sequences):     {total_fa_frames}")
        print(f"    Total duration:                      {total_duration_h * 60:.1f} min")
        print(f"    Aggregate FA/h:                      {total_fa_per_h:.2f}")
        print("=" * 62 + "\n")

if __name__ == "__main__":
    main()
