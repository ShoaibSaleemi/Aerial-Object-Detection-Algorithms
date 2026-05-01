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
        # Each entry: (KalmanBoxTracker, class_id, lstm_hidden_state)
        self.tracks: list[tuple[KalmanBoxTracker, int, object]] = []
        self.lstm   = TrajectoryLSTM().to(lstm_device).eval()
        self.device = lstm_device
        # track_id → list of (int cx, int cy) pixel centres
        self.trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # ── public ───────────────────────────────────────────────────────────────
    def update(
        self,
        detections: list[tuple[list[float], int]],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[list[float], int, int, list[float] | None]]:
        """
        Process one frame.

        detections : list of (box_xyxy, class_id)
        Returns    : list of (box_xyxy, track_id, class_id, lstm_pred_xyxy_or_None)
        """
        # ── Kalman predict ──
        preds_xyxy = [xywh_to_xyxy(trk.predict()) for trk, _, __ in self.tracks]

        # ── Greedy IoU matching ──
        n_t, n_d = len(preds_xyxy), len(detections)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        pairs: list[tuple[int, int]] = []

        if n_t and n_d:
            cost = np.zeros((n_t, n_d), dtype=np.float32)
            for ti, pxy in enumerate(preds_xyxy):
                for di, (dxy, _) in enumerate(detections):
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
            dxy, dcls = detections[di]
            xywh = xyxy_to_xywh(dxy)
            trk, _, hidden = self.tracks[ti]
            trk.update(xywh)
            hidden = self._lstm_step(xywh, frame_w, frame_h, hidden)
            self.tracks[ti] = (trk, dcls, hidden)

        # ── Increment lost counter for unmatched tracks ──
        for ti, (trk, cls, hid) in enumerate(self.tracks):
            if ti not in matched_t:
                trk.lost += 1

        # ── Spawn new tracks for unmatched detections ──
        for di, (dxy, dcls) in enumerate(detections):
            if di not in matched_d:
                xywh = xyxy_to_xywh(dxy)
                self.tracks.append((KalmanBoxTracker(xywh), dcls, None))

        # ── Prune dead tracks ──
        self.tracks = [(t, c, h) for t, c, h in self.tracks if t.lost <= MAX_LOST]

        # ── Build output and update trails ──
        results: list[tuple[list[float], int, int, list[float] | None]] = []
        for trk, cls, hidden in self.tracks:
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
            results.append((box_xyxy, trk.id, cls, lstm_pred))

        # ── Prune trails for deleted tracks ──
        live_ids = {trk.id for trk, _, __ in self.tracks}
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


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_tracks(
    frame: np.ndarray,
    track_results: list,
    trails: dict,
):
    for box_xyxy, track_id, cls_int, lstm_pred in track_results:
        label = CLASS_NAMES[cls_int] if cls_int < len(CLASS_NAMES) else "unknown"
        color = COLORS[label]

        # Bounding box.
        x1, y1, x2, y2 = (int(v) for v in box_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label without track ID.
        text = f"{label}"
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
def load_gt_labels(video_path: Path):
    """Load <stem>.json GT alongside the video if present."""
    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get("exist", []), data.get("gt_rect", [])


# ── Questionary helpers (same as inference_video.py) ─────────────────────────
def choose_run_folder() -> str:
    detect_root_dir = PROJECT_ROOT / "runs" / "detect"
    available_runs  = sorted(
        [p.name for p in detect_root_dir.iterdir() if p.is_dir()]
    )

    if not available_runs:
        raise ValueError(f"No folders found in {detect_root_dir}")

    if len(sys.argv) > 1:
        run_name = sys.argv[1]
        if run_name not in available_runs:
            raise ValueError(
                f"Unknown folder '{run_name}'. "
                f"Choose one from runs/detect: {', '.join(available_runs)}"
            )
        return run_name

    run_name = questionary.select(
        "Choose a folder from runs/detect:",
        choices=available_runs,
    ).ask()
    if not run_name:
        raise ValueError("No folder selected from runs/detect")
    return run_name


def choose_video_file() -> Path:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_paths:
        raise ValueError(f"No .mp4 video files found in {VIDEO_DIR}")

    if len(sys.argv) > 2:
        video_name = sys.argv[2]
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
        raise ValueError("No video selected from dataset/test/videos")
    return VIDEO_DIR / video_name


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    run_name   = choose_run_folder()
    video_path = choose_video_file()

    detect_run_dir = PROJECT_ROOT / "runs" / "detect" / run_name
    model_path     = detect_run_dir / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_path = PROJECT_ROOT / "runs" / "detect" / "inference" / f"{video_path.stem}_{run_name}_tracking.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve LSTM / torch device.
    if DEVICE:
        lstm_device = torch.device("cuda:0" if DEVICE == "0" else DEVICE)
    else:
        lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model   = YOLO(str(model_path))
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

    print(f"{'Loaded model:':<22}{model_path}")
    print(f"{'Input video:':<22}{video_path}")
    print(f"{'Saving output to:':<22}{output_path}")

    # Ground-truth labels (optional)
    gt_data   = load_gt_labels(video_path)
    has_gt    = gt_data is not None
    gt_exist: list = []
    gt_rect:  list = []
    if has_gt:
        gt_exist, gt_rect = gt_data
        print(f"{'GT labels:':<22}{video_path.stem}.json  ({len(gt_exist)} frames)")

    print()

    frame_ious:     list[float] = []
    frame_dists:    list[float] = []
    covered_frames: int = 0
    exist1_frames:  int = 0

    processed  = 0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        predict_kwargs = {
            "source": frame,
            "conf":   CONF_THRESH,
            "imgsz":  IMG_SIZE,
            "batch":  1,
            "save":   False,
            "show":   False,
            "verbose": False,
        }
        if DEVICE:
            predict_kwargs["device"] = DEVICE

        result = model.predict(**predict_kwargs)
        r = result[0] if isinstance(result, list) else result

        # Build detection list with open-set class remap (same as inference_video.py).
        detections: list[tuple[list[float], int]] = []
        for box, cls_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            cls_raw = int(cls_id)
            cls_int = cls_raw if cls_raw in (0, 1) else 2
            detections.append((box.tolist(), cls_int))

        track_results = tracker.update(detections, width, height)
        draw_tracks(frame, track_results, tracker.trails)

        # ── GT overlay + per-frame evaluation ──────────────────────────────
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
                for box_xyxy, _tid, _cls, _lstm in track_results:
                    iou_val = compute_iou(box_xyxy, gt_xyxy)
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

    # ── Evaluation summary ──────────────────────────────────────────────────
    if has_gt and exist1_frames > 0:
        iou_arr  = np.array(frame_ious,  dtype=np.float32)
        dist_arr = np.array(frame_dists, dtype=np.float32)

        thr_iou  = np.linspace(0.0, 1.0, 101)
        success  = np.array([(iou_arr >= t).mean() for t in thr_iou], dtype=np.float32)
        auc      = float(np.trapezoid(success, thr_iou))
        sr50     = float((iou_arr >= 0.5).mean())

        thr_dist  = np.arange(0, 51, dtype=np.float32)
        precision = np.array([(dist_arr <= t).mean() for t in thr_dist], dtype=np.float32)
        prec20    = float((dist_arr <= 20.0).mean())
        coverage  = covered_frames / exist1_frames
        mean_iou  = float(iou_arr.mean())

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
        plot_path = output_path.parent / f"{video_path.stem}_{run_name}_eval.png"
        fig.savefig(str(plot_path), dpi=120)
        plt.close(fig)
        print(f"\n  Eval plot → {plot_path}")

        csv_path = output_path.parent / f"{video_path.stem}_{run_name}_eval.csv"
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
