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
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["bird", "drone", "unknown"]
VIDEO_DIR   = PROJECT_ROOT / "dataset" / "test" / "videos"
CONF_THRESH = 0.7           # per-model detection threshold before fusion
IMG_SIZE    = 640
DEVICE      = ""             # "cpu", "0", etc.; empty = auto

COLORS = {
    "bird":    (0, 255, 0),      # green
    "drone":   (0, 0, 255),      # red
    "unknown": (0, 165, 255),    # orange
}

# ── WBF parameters ────────────────────────────────────────────────────────────
FUSION_IOU_THRESH       = 0.50
MIN_MODEL_SUPPORT       = 3
KNOWN_FUSED_CONF_THRESH = 0.55
SCORE_MARGIN_THRESH     = 0.20
DISAGREEMENT_RATIO_THRESH = 0.55

# Ensemble model list: (name, path_to_weights)
MODELS = [
    ("yolo8n",  PROJECT_ROOT / "runs" / "detect" / "yolo8n"  / "weights" / "best.pt"),
    ("yolo9t",  PROJECT_ROOT / "runs" / "detect" / "yolo9t"  / "weights" / "best.pt"),
    ("yolo10n", PROJECT_ROOT / "runs" / "detect" / "yolo10n" / "weights" / "best.pt"),
]

# Per-model per-class weighting for WBF.
MODEL_WEIGHTS = {
    "yolo8n":  {"bird": 1.000, "drone": 1.000, "unknown": 1.000},
    "yolo9t":  {"bird": 1.000, "drone": 1.000, "unknown": 1.000},
    "yolo10n": {"bird": 1.000, "drone": 1.000, "unknown": 1.000},
}

# ── Tracking parameters ───────────────────────────────────────────────────────
MAX_LOST   = 10   # frames to keep a lost track alive
MIN_HITS   = 2    # frames before a new track is drawn
IOU_THRESH_TRACK = 0.30  # greedy matching IoU threshold
SEQ_LEN    = 8    # LSTM history window (frames)
TRAIL_LEN  = 30   # trail length in frames


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
            "conf":   CONF_THRESH,
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

    output_dir = PROJECT_ROOT / "runs" / "detect" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_wbf_tracking.mp4"

    # Torch device for LSTM
    if DEVICE:
        lstm_device = torch.device("cuda:0" if DEVICE == "0" else DEVICE)
    else:
        lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ensemble models
    print("Loading ensemble models...")
    loaded_models: list[tuple[str, YOLO, dict]] = []
    for model_name, model_path in MODELS:
        if not model_path.exists():
            print(f"  [SKIP] {model_name}: not found at {model_path}")
            continue
        model     = YOLO(str(model_path))
        class_map = build_model_class_map(model)
        loaded_models.append((model_name, model, class_map))
        print(f"  [OK]   {model_name}")

    if not loaded_models:
        raise ValueError("No ensemble models were loaded. Check MODELS paths.")

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
    frame_dists:    list[float] = []  # centre distance per exist=1 frame
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
                best_dist = float("inf")
                for box_xyxy, _tid, _cls, _conf, _lstm in track_results:
                    iou_val = _iou(box_xyxy, gt_xyxy)
                    if iou_val > best_iou:
                        best_iou  = iou_val
                        tcx = (box_xyxy[0] + box_xyxy[2]) / 2.0
                        tcy = (box_xyxy[1] + box_xyxy[3]) / 2.0
                        best_dist = float(np.hypot(tcx - gt_cx, tcy - gt_cy))

                frame_ious.append(best_iou)
                frame_dists.append(best_dist)
                if track_results:
                    covered_frames += 1

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

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  Evaluation  ({exist1_frames} present frames / {processed} total)")
        print(sep)
        print(f"  {'Mean IoU:':<32}{mean_iou * 100:.2f}%")
        print(f"  {'Success Rate  @IoU≥0.5:':<32}{sr50 * 100:.2f}%")
        print(f"  {'AUC  (success curve 0→1):':<32}{auc * 100:.2f}%")
        print(f"  {'Precision  @20 px:':<32}{prec20 * 100:.2f}%")
        print(f"  {'Coverage:':<32}{coverage * 100:.2f}%  ({covered_frames}/{exist1_frames})")
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

        csv_path = output_dir / f"{video_path.stem}_wbf_eval.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value", "value_percent", "note"])
            writer.writerow(["mean_iou", mean_iou, mean_iou * 100.0, ""])
            writer.writerow(["success_rate_iou_ge_0_5", sr50, sr50 * 100.0, "IoU >= 0.5"])
            writer.writerow(["auc_success_curve_0_to_1", auc, auc * 100.0, ""])
            writer.writerow(["precision_at_20px", prec20, prec20 * 100.0, "center distance <= 20"])
            writer.writerow(["coverage", coverage, coverage * 100.0, f"{covered_frames}/{exist1_frames}"])
            writer.writerow(["covered_frames", covered_frames, "", ""])
            writer.writerow(["present_frames", exist1_frames, "", ""])
            writer.writerow(["processed_frames", processed, "", ""])
        print(f"  Eval csv  → {csv_path}")


if __name__ == "__main__":
    main()
