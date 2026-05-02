"""
Random-search hyperparameter tuning for the WBF + Kalman tracking pipeline.

Loads ensemble models once, then for each trial patches the module-level
globals in inference_video_wbf_tracking and runs the full evaluation loop
across ALL validation sequences.  Scores are averaged over sequences.
No output video is written during tuning.

Optimises a composite score (averaged across validation sequences):
    score = 0.40 * AUC  +  0.30 * SR@IoU≥0.5  +  0.30 * Prec@20px

Usage:
    python "tools/tune_wbf_tracking.py"
    python "tools/tune_wbf_tracking.py" --trials 100 --seed 42
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ── resolve paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = PROJECT_ROOT / "scripts" / "model testing" / "inference_video_wbf_tracking.py"
if not INFERENCE_PATH.exists():
    raise FileNotFoundError(f"Inference script not found: {INFERENCE_PATH}")

_spec = importlib.util.spec_from_file_location("inference_video_wbf_tracking", INFERENCE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot create import spec for: {INFERENCE_PATH}")
inf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inf)

# ── config ───────────────────────────────────────────────────────────────────
# Validation sequences directory — each sub-folder contains visible.mp4 + visible.json
VAL_VIDEOS_DIR = PROJECT_ROOT / "dataset" / "validation" / "videos"

# Tuning output
BEST_JSON  = PROJECT_ROOT / "runs" / "detect" / "tune_wbf" / "best_params.json"

# Weights for composite objective (must sum to 1.0)
W_AUC    = 0.40
W_SR50   = 0.30
W_PREC20 = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Load GT once
# ─────────────────────────────────────────────────────────────────────────────
def load_gt(json_path: Path) -> tuple[list, list]:
    with open(json_path) as f:
        data = json.load(f)
    return data["exist"], data["gt_rect"]


# ─────────────────────────────────────────────────────────────────────────────
# Load ensemble models once (expensive; reuse across trials)
# ─────────────────────────────────────────────────────────────────────────────
def load_models() -> list:
    loaded = []
    for model_name, model_path in inf.MODELS:
        if not model_path.exists():
            print(f"  [SKIP] {model_name}: not found at {model_path}")
            continue
        from ultralytics import YOLO  # noqa: PLC0415
        model     = YOLO(str(model_path))
        class_map = inf.build_model_class_map(model)
        loaded.append((model_name, model, class_map))
        print(f"  [OK]   {model_name}")
    if not loaded:
        raise RuntimeError("No ensemble models loaded. Check MODELS paths.")
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ─────────────────────────────────────────────────────────────────────────────
def _patch_globals(params: dict) -> None:
    """Write trial parameters into the inference module's global namespace."""
    for key, val in params.items():
        setattr(inf, key, val)


def _make_kalman_init(r_noise: float, q_vel: float, p_vel: float):
    """Return a patched __init__ for KalmanBoxTracker with tunable noise."""
    original_init = inf.KalmanBoxTracker.__init__

    def _new_init(self, xywh):
        original_init(self, xywh)
        # Override noise matrices after base init
        self.R         = np.eye(4, dtype=np.float32) * r_noise
        self.Q         = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= q_vel
        self.P         = np.eye(8, dtype=np.float32)
        self.P[4:, 4:] *= p_vel

    return _new_init


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop (no video writing)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(loaded_models: list, video_path: Path, gt_exist: list, gt_rect: list,
             lstm_device: torch.device, seq_idx: int | None = None,
             total_seq: int | None = None,
             sim_start_time: float | None = None) -> dict:
    """Run the full pipeline on a single video and return evaluation metrics."""

    # Reset track ID counter so each trial starts fresh
    inf.KalmanBoxTracker._next_id = 0

    tracker = inf.MultiObjectTracker(lstm_device=lstm_device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_ious:     list[float] = []
    frame_dists:    list[float] = []
    covered_frames: int = 0
    exist1_frames:  int = 0
    processed:      int = 0
    last_status_len = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fused      = inf.run_wbf_on_frame(loaded_models, frame)
        detections = [(d["box"], d["class_id"], d["confidence"]) for d in fused]
        track_results = tracker.update(detections, width, height)

        if processed < len(gt_exist) and gt_exist[processed] == 1:
            rx, ry, rw, rh = gt_rect[processed]
            gt_xyxy = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
            gt_cx   = rx + rw / 2.0
            gt_cy   = ry + rh / 2.0
            exist1_frames += 1

            best_iou  = 0.0
            best_dist = float("inf")
            for box_xyxy, _tid, _cls, _conf, _lstm in track_results:
                iou_val = inf._iou(box_xyxy, gt_xyxy)
                if iou_val > best_iou:
                    best_iou  = iou_val
                    tcx = (box_xyxy[0] + box_xyxy[2]) / 2.0
                    tcy = (box_xyxy[1] + box_xyxy[3]) / 2.0
                    best_dist = float(np.hypot(tcx - gt_cx, tcy - gt_cy))

            frame_ious.append(best_iou)
            frame_dists.append(best_dist if best_dist != float("inf") else 9999.0)
            if track_results:
                covered_frames += 1

        processed += 1
        elapsed_total = (time.time() - sim_start_time) if sim_start_time is not None else 0.0
        t_min, t_sec = divmod(int(elapsed_total), 60)
        video_prefix = ""
        if seq_idx is not None and total_seq is not None:
            video_prefix = f"Video {seq_idx}/{total_seq} | "
        if total_frames > 0:
            pct = processed / total_frames * 100
            status = (
                f"    {video_prefix}Frame: {processed}/{total_frames} ({pct:.1f}%)"
                f" | Elapsed: {t_min}:{t_sec:02d}"
            )
        else:
            status = f"    {video_prefix}Frame: {processed} | Elapsed: {t_min}:{t_sec:02d}"
        padded = status.ljust(last_status_len)
        print(f"\r{padded}", end="", flush=True)
        last_status_len = max(last_status_len, len(status))

    if last_status_len > 0:
        print("\r" + " " * last_status_len, end="")
    print("\r", end="")
    cap.release()

    if exist1_frames == 0:
        return {"auc": 0.0, "sr50": 0.0, "prec20": 0.0, "coverage": 0.0, "score": 0.0}

    iou_arr  = np.array(frame_ious,  dtype=np.float32)
    dist_arr = np.array(frame_dists, dtype=np.float32)

    thr_iou = np.linspace(0.0, 1.0, 101)
    success = np.array([(iou_arr >= t).mean() for t in thr_iou], dtype=np.float32)
    auc     = float(np.trapezoid(success, thr_iou))
    sr50    = float((iou_arr >= 0.5).mean())
    prec20  = float((dist_arr <= 20.0).mean())
    coverage = covered_frames / exist1_frames

    score = W_AUC * auc + W_SR50 * sr50 + W_PREC20 * prec20

    return {
        "auc":      auc,
        "sr50":     sr50,
        "prec20":   prec20,
        "coverage": coverage,
        "score":    score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sequence evaluation — average metrics across all validation sequences
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_all(loaded_models: list,
                 sequences: list[tuple[Path, list, list]],
                 lstm_device: torch.device,
                 sim_start_time: float | None = None) -> dict:
    """Evaluate across all sequences and return mean metrics."""
    keys = ("auc", "sr50", "prec20", "coverage", "score")
    totals = {k: 0.0 for k in keys}
    valid  = 0

    n_seq = len(sequences)
    for i, (video_path, gt_exist, gt_rect) in enumerate(sequences, 1):
        try:
            m = evaluate(
                loaded_models,
                video_path,
                gt_exist,
                gt_rect,
                lstm_device,
                seq_idx=i,
                total_seq=n_seq,
                sim_start_time=sim_start_time,
            )
        except Exception as exc:
            print(f"  [WARN] Skipping sequence {i}: {exc}")
            continue
        for k in keys:
            totals[k] += m[k]
        valid += 1

    if valid == 0:
        return {k: 0.0 for k in keys}

    return {k: totals[k] / valid for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Random search helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sample_params(rng: np.random.Generator, n_models: int) -> dict:
    """Sample one random parameter set from the same ranges used before."""
    w_yolo8n  = float(rng.uniform(0.5, 2.0))
    w_yolo9t  = float(rng.uniform(0.5, 2.0))
    w_yolo10n = float(rng.uniform(0.5, 2.0))

    return {
        "CONF_THRESH":               float(rng.uniform(0.30, 0.85)),
        "FUSION_IOU_THRESH":         float(rng.uniform(0.25, 0.75)),
        "MIN_MODEL_SUPPORT":         int(rng.integers(1, n_models + 1)),
        "KNOWN_FUSED_CONF_THRESH":   float(rng.uniform(0.20, 0.80)),
        "SCORE_MARGIN_THRESH":       float(rng.uniform(0.05, 0.60)),
        "DISAGREEMENT_RATIO_THRESH": float(rng.uniform(0.20, 0.90)),
        "MAX_LOST":                  int(rng.integers(1, 31)),
        "MIN_HITS":                  int(rng.integers(1, 6)),
        "IOU_THRESH_TRACK":          float(rng.uniform(0.10, 0.60)),
        "SEQ_LEN":                   int(rng.integers(3, 21)),
        "kalman_R":                  float(rng.uniform(0.5, 20.0)),
        "kalman_Q_vel":              float(10 ** rng.uniform(np.log10(1e-4), np.log10(0.50))),
        "kalman_P_vel":              float(10 ** rng.uniform(np.log10(10.0), np.log10(5000.0))),
        "w_yolo8n":                  w_yolo8n,
        "w_yolo9t":                  w_yolo9t,
        "w_yolo10n":                 w_yolo10n,
        "MODEL_WEIGHTS": {
            "yolo8n":  {"bird": w_yolo8n,  "drone": w_yolo8n,  "unknown": w_yolo8n},
            "yolo9t":  {"bird": w_yolo9t,  "drone": w_yolo9t,  "unknown": w_yolo9t},
            "yolo10n": {"bird": w_yolo10n, "drone": w_yolo10n, "unknown": w_yolo10n},
        },
    }


def _evaluate_params(loaded_models: list,
                     sequences: list[tuple[Path, list, list]],
                     lstm_device: torch.device,
                     params: dict,
                     trial_idx: int,
                     total_trials: int,
                     sim_start_time: float | None = None) -> dict:
    print(f"\n[Trial {trial_idx}/{total_trials}]", flush=True)

    _patch_globals({
        "CONF_THRESH":               params["CONF_THRESH"],
        "FUSION_IOU_THRESH":         params["FUSION_IOU_THRESH"],
        "MIN_MODEL_SUPPORT":         params["MIN_MODEL_SUPPORT"],
        "KNOWN_FUSED_CONF_THRESH":   params["KNOWN_FUSED_CONF_THRESH"],
        "SCORE_MARGIN_THRESH":       params["SCORE_MARGIN_THRESH"],
        "DISAGREEMENT_RATIO_THRESH": params["DISAGREEMENT_RATIO_THRESH"],
        "MAX_LOST":                  params["MAX_LOST"],
        "MIN_HITS":                  params["MIN_HITS"],
        "IOU_THRESH_TRACK":          params["IOU_THRESH_TRACK"],
        "SEQ_LEN":                   params["SEQ_LEN"],
        "MODEL_WEIGHTS":             params["MODEL_WEIGHTS"],
    })

    inf.KalmanBoxTracker.__init__ = _make_kalman_init(
        params["kalman_R"],
        params["kalman_Q_vel"],
        params["kalman_P_vel"],
    )

    metrics = evaluate_all(
        loaded_models,
        sequences,
        lstm_device,
        sim_start_time=sim_start_time,
    )
    print(f"  Trial score: {metrics['score']:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Random-search tuning for WBF+Kalman tracker")
    parser.add_argument("--trials", type=int, default=50, help="Number of random-search trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not VAL_VIDEOS_DIR.is_dir():
        raise FileNotFoundError(f"Validation videos dir not found: {VAL_VIDEOS_DIR}")

    print("=" * 60)
    print("  WBF + Kalman Tracking — Random Search Tuner")
    print("=" * 60)
    print(f"  Val dir: {VAL_VIDEOS_DIR}")
    print(f"  Trials : {args.trials}")
    print(f"  Seed   : {args.seed}")
    print()

    print("Discovering validation sequences...")
    sequences: list[tuple[Path, list, list]] = []
    for seq_dir in sorted(VAL_VIDEOS_DIR.iterdir()):
        video_path = seq_dir / "visible.mp4"
        gt_path    = seq_dir / "visible.json"
        if not video_path.exists() or not gt_path.exists():
            print(f"  [SKIP] {seq_dir.name}: missing video or GT")
            continue
        gt_exist, gt_rect = load_gt(gt_path)
        sequences.append((video_path, gt_exist, gt_rect))
    if not sequences:
        raise RuntimeError("No valid validation sequences found.")
    print(f"  {len(sequences)} sequences loaded")

    print("\nLoading ensemble models...")
    loaded_models = load_models()

    # LSTM / Kalman device
    lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {lstm_device}")

    rng = np.random.default_rng(args.seed)

    # Use previous defaults as trial 1, random for remaining trials
    default_params = {
        "CONF_THRESH":               0.70,
        "FUSION_IOU_THRESH":         0.50,
        "MIN_MODEL_SUPPORT":         3,
        "KNOWN_FUSED_CONF_THRESH":   0.55,
        "SCORE_MARGIN_THRESH":       0.20,
        "DISAGREEMENT_RATIO_THRESH": 0.55,
        "MAX_LOST":                  10,
        "MIN_HITS":                  2,
        "IOU_THRESH_TRACK":          0.30,
        "SEQ_LEN":                   8,
        "kalman_R":                  4.0,
        "kalman_Q_vel":              0.01,
        "kalman_P_vel":              1000.0,
        "w_yolo8n":                  1.0,
        "w_yolo9t":                  1.0,
        "w_yolo10n":                 1.0,
    }
    default_params["MODEL_WEIGHTS"] = {
        "yolo8n":  {"bird": 1.0, "drone": 1.0, "unknown": 1.0},
        "yolo9t":  {"bird": 1.0, "drone": 1.0, "unknown": 1.0},
        "yolo10n": {"bird": 1.0, "drone": 1.0, "unknown": 1.0},
    }

    print(f"\nStarting optimisation ({args.trials} trials)...\n")

    best = None
    history = []
    sim_start_time = time.time()
    for trial_idx in range(1, args.trials + 1):
        params = default_params if trial_idx == 1 else _sample_params(rng, len(loaded_models))
        metrics = _evaluate_params(
            loaded_models=loaded_models,
            sequences=sequences,
            lstm_device=lstm_device,
            params=params,
            trial_idx=trial_idx,
            total_trials=args.trials,
            sim_start_time=sim_start_time,
        )

        record = {
            "trial": trial_idx,
            "score": metrics["score"],
            "metrics": {k: metrics[k] for k in ("auc", "sr50", "prec20", "coverage")},
            "params": {k: v for k, v in params.items() if k != "MODEL_WEIGHTS"},
        }
        history.append(record)

        if best is None or record["score"] > best["score"]:
            best = record
            print(f"  New best: {best['score']:.4f}")

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Best trial")
    print("=" * 60)
    if best is None:
        raise RuntimeError("No valid trial results.")

    print(f"  Score    : {best['score']:.4f}")
    print(f"  AUC      : {best['metrics']['auc']:.4f}")
    print(f"  SR@IoU≥.5: {best['metrics']['sr50']:.4f}")
    print(f"  Prec@20px: {best['metrics']['prec20']:.4f}")
    print(f"  Coverage : {best['metrics']['coverage']:.4f}")
    print("\n  Parameters:")
    for k, v in best["params"].items():
        print(f"    {k:<32} {v}")

    # Save best params to JSON
    BEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "score":    best["score"],
        "metrics":  best["metrics"],
        "params":   best["params"],
        "history":  history,
    }
    with open(BEST_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Best params saved → {BEST_JSON}")


if __name__ == "__main__":
    main()
