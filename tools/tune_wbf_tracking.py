"""
Optuna hyperparameter tuning for the WBF + Kalman tracking pipeline.

Loads ensemble models once, then for each trial patches the module-level
globals in inference_video_wbf_tracking and runs the full evaluation loop
across ALL validation sequences.  Scores are averaged over sequences.
No output video is written during tuning.

Optimises a composite score (averaged across validation sequences):
    score = 0.40 * AUC  +  0.30 * SR@IoU≥0.5  +  0.30 * Prec@20px

Usage:
    python "scripts/model testing/tune_wbf_tracking.py"
    python "scripts/model testing/tune_wbf_tracking.py" --trials 100
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import optuna
import torch

# ── resolve paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR  = Path(__file__).resolve().parent

# Add the "model testing" folder to sys.path so the sibling module is importable
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import inference_video_wbf_tracking as inf  # noqa: E402

# ── config ───────────────────────────────────────────────────────────────────
# Validation sequences directory — each sub-folder contains visible.mp4 + visible.json
VAL_VIDEOS_DIR = PROJECT_ROOT / "dataset" / "validation" / "videos"

# Optuna study output
STUDY_DB   = PROJECT_ROOT / "runs" / "detect" / "tune_wbf" / "study.db"
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
             lstm_device: torch.device) -> dict:
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
    start_time = time.time()

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
        if total_frames > 0:
            pct = processed / total_frames * 100
            print(
                f"    Frame: {processed}/{total_frames} ({pct:.1f}%)",
                end="\r",
            )
        else:
            print(
                f"    Frame: {processed}",
                end="\r",
            )

    elapsed = time.time() - start_time
    print()  # newline after frame progress
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
        "elapsed":  elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sequence evaluation — average metrics across all validation sequences
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_all(loaded_models: list,
                 sequences: list[tuple[Path, list, list]],
                 lstm_device: torch.device) -> dict:
    """Evaluate across all sequences and return mean metrics."""
    keys = ("auc", "sr50", "prec20", "coverage", "score")
    totals = {k: 0.0 for k in keys}
    valid  = 0

    n_seq = len(sequences)
    for i, (video_path, gt_exist, gt_rect) in enumerate(sequences, 1):
        print(f"  Sequence {i}/{n_seq}", end="", flush=True)
        try:
            m = evaluate(loaded_models, video_path, gt_exist, gt_rect, lstm_device)
        except Exception as exc:
            print(f"  [WARN] Skipping sequence {i}: {exc}")
            continue
        mins, secs = divmod(int(m["elapsed"]), 60)
        print(f"  ({mins}:{secs:02d})")
        for k in keys:
            totals[k] += m[k]
        valid += 1

    if valid == 0:
        return {k: 0.0 for k in keys}

    return {k: totals[k] / valid for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(loaded_models: list,
                   sequences: list[tuple[Path, list, list]],
                   lstm_device: torch.device):

    def objective(trial: optuna.Trial) -> float:
        # ── WBF / detection params ─────────────────────────────────────────
        conf_thresh              = trial.suggest_float("CONF_THRESH",              0.30, 0.85)
        fusion_iou_thresh        = trial.suggest_float("FUSION_IOU_THRESH",        0.25, 0.75)
        min_model_support        = trial.suggest_int  ("MIN_MODEL_SUPPORT",        1,    len(loaded_models))
        known_fused_conf_thresh  = trial.suggest_float("KNOWN_FUSED_CONF_THRESH",  0.20, 0.80)
        score_margin_thresh      = trial.suggest_float("SCORE_MARGIN_THRESH",      0.05, 0.60)
        disagreement_ratio_thresh = trial.suggest_float("DISAGREEMENT_RATIO_THRESH", 0.20, 0.90)

        # ── Tracking params ────────────────────────────────────────────────
        max_lost        = trial.suggest_int  ("MAX_LOST",        1,  30)
        min_hits        = trial.suggest_int  ("MIN_HITS",        1,  5)
        iou_thresh_track = trial.suggest_float("IOU_THRESH_TRACK", 0.10, 0.60)
        seq_len         = trial.suggest_int  ("SEQ_LEN",         3,  20)

        # ── Kalman noise params ────────────────────────────────────────────
        r_noise = trial.suggest_float("kalman_R",     0.5,  20.0)
        q_vel   = trial.suggest_float("kalman_Q_vel", 1e-4, 0.50,  log=True)
        p_vel   = trial.suggest_float("kalman_P_vel", 10.0, 5000.0, log=True)

        # ── Model weights (symmetric per class for simplicity) ─────────────
        w_yolo8n  = trial.suggest_float("w_yolo8n",  0.5, 2.0)
        w_yolo9t  = trial.suggest_float("w_yolo9t",  0.5, 2.0)
        w_yolo10n = trial.suggest_float("w_yolo10n", 0.5, 2.0)

        # ── Apply globals ──────────────────────────────────────────────────
        _patch_globals({
            "CONF_THRESH":               conf_thresh,
            "FUSION_IOU_THRESH":         fusion_iou_thresh,
            "MIN_MODEL_SUPPORT":         min_model_support,
            "KNOWN_FUSED_CONF_THRESH":   known_fused_conf_thresh,
            "SCORE_MARGIN_THRESH":       score_margin_thresh,
            "DISAGREEMENT_RATIO_THRESH": disagreement_ratio_thresh,
            "MAX_LOST":                  max_lost,
            "MIN_HITS":                  min_hits,
            "IOU_THRESH_TRACK":          iou_thresh_track,
            "SEQ_LEN":                   seq_len,
            "MODEL_WEIGHTS": {
                "yolo8n":  {"bird": w_yolo8n,  "drone": w_yolo8n,  "unknown": w_yolo8n},
                "yolo9t":  {"bird": w_yolo9t,  "drone": w_yolo9t,  "unknown": w_yolo9t},
                "yolo10n": {"bird": w_yolo10n, "drone": w_yolo10n, "unknown": w_yolo10n},
            },
        })

        # ── Patch Kalman __init__ ──────────────────────────────────────────
        inf.KalmanBoxTracker.__init__ = _make_kalman_init(r_noise, q_vel, p_vel)

        # ── Run evaluation across all validation sequences ─────────────────
        metrics = evaluate_all(loaded_models, sequences, lstm_device)

        trial.set_user_attr("auc",      metrics["auc"])
        trial.set_user_attr("sr50",     metrics["sr50"])
        trial.set_user_attr("prec20",   metrics["prec20"])
        trial.set_user_attr("coverage", metrics["coverage"])

        return metrics["score"]

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for WBF+Kalman tracker")
    parser.add_argument("--trials",   type=int, default=50,  help="Number of Optuna trials")
    parser.add_argument("--jobs",     type=int, default=1,   help="Parallel jobs (1 = sequential)")
    parser.add_argument("--sampler",  choices=["tpe", "random"], default="tpe")
    parser.add_argument("--no-db",    action="store_true",   help="Don't persist study to SQLite")
    args = parser.parse_args()

    if not VAL_VIDEOS_DIR.is_dir():
        raise FileNotFoundError(f"Validation videos dir not found: {VAL_VIDEOS_DIR}")

    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  WBF + Kalman Tracking — Optuna Hyperparameter Tuner")
    print("=" * 60)
    print(f"  Val dir: {VAL_VIDEOS_DIR}")
    print(f"  Trials : {args.trials}")
    print(f"  Sampler: {args.sampler}")
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

    # ── Optuna study ──────────────────────────────────────────────────────
    sampler = (optuna.samplers.TPESampler(seed=42)
               if args.sampler == "tpe"
               else optuna.samplers.RandomSampler(seed=42))

    storage = None if args.no_db else f"sqlite:///{STUDY_DB}"

    study = optuna.create_study(
        study_name="wbf_kalman_tuning",
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(loaded_models, sequences, lstm_device)

    # Seed with default params as starting trial
    study.enqueue_trial({
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
    })

    print(f"\nStarting optimisation ({args.trials} trials)...\n")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=True)

    # ── Results ───────────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "=" * 60)
    print("  Best trial")
    print("=" * 60)
    print(f"  Score    : {best.value:.4f}")
    print(f"  AUC      : {best.user_attrs.get('auc', '?'):.4f}")
    print(f"  SR@IoU≥.5: {best.user_attrs.get('sr50', '?'):.4f}")
    print(f"  Prec@20px: {best.user_attrs.get('prec20', '?'):.4f}")
    print(f"  Coverage : {best.user_attrs.get('coverage', '?'):.4f}")
    print("\n  Parameters:")
    for k, v in best.params.items():
        print(f"    {k:<32} {v}")

    # Save best params to JSON
    BEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "score":    best.value,
        "metrics":  {k: best.user_attrs.get(k) for k in ("auc", "sr50", "prec20", "coverage")},
        "params":   best.params,
    }
    with open(BEST_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Best params saved → {BEST_JSON}")

    if storage:
        print(f"  Study DB           → {STUDY_DB}")
        print("  (Resume with the same command — completed trials are skipped)")


if __name__ == "__main__":
    main()
