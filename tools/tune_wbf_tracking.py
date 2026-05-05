"""
Bayesian optimization hyperparameter tuning for the WBF + Kalman tracking pipeline.

Loads ensemble models once, then uses Optuna's TPE sampler to tune:
  - CONF_THRESH, FUSION_IOU_THRESH
  - MIN_MODEL_SUPPORT, KNOWN_FUSED_CONF_THRESH, SCORE_MARGIN_THRESH, DISAGREEMENT_RATIO_THRESH
  - MAX_LOST, MIN_HITS, IOU_THRESH_TRACK, SEQ_LEN
  - Kalman noise parameters (R, Q_vel, P_vel)
  - Per-model weights

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
import optuna
from optuna.samplers import TPESampler

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
OPTUNA_DB  = PROJECT_ROOT / "runs" / "detect" / "tune_wbf" / "optuna_tracking_study.db"

# Weights for composite objective (must sum to 1.0)
W_AUC    = 0.40
W_SR50   = 0.30
W_PREC20 = 0.30

_INLINE_STATUS_LEN = 0


def _print_inline_status(status: str) -> None:
    """Print status on one updating terminal line, clearing leftovers."""
    global _INLINE_STATUS_LEN
    if sys.stdout.isatty():
        padded = status.ljust(_INLINE_STATUS_LEN)
        print(f"\r{padded}", end="", flush=True)
        _INLINE_STATUS_LEN = len(status)
    else:
        print(status, flush=True)


def _finish_inline_status_line() -> None:
    """Move cursor to the next line after inline updates."""
    global _INLINE_STATUS_LEN
    if sys.stdout.isatty():
        print()
    _INLINE_STATUS_LEN = 0


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
             sim_start_time: float | None = None,
             label: str = "") -> dict:
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
        t_hour, t_rem = divmod(int(elapsed_total), 3600)
        t_min, t_sec  = divmod(t_rem, 60)
        elapsed_str   = f"{t_hour}:{t_min:02d}:{t_sec:02d}"
        trial_prefix = f"{label} " if label else ""
        video_prefix = ""
        if seq_idx is not None and total_seq is not None:
            video_prefix = f"Video {seq_idx}/{total_seq} | "
        if total_frames > 0:
            pct = processed / total_frames * 100
            status = (
                f"    {trial_prefix}{video_prefix}Frame: {processed}/{total_frames} ({pct:.1f}%)"
                f" | Elapsed: {elapsed_str}"
            )
        else:
            status = f"    {trial_prefix}{video_prefix}Frame: {processed} | Elapsed: {elapsed_str}"
        _print_inline_status(status)

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
                 sim_start_time: float | None = None,
                 trial_label: str = "") -> dict:
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
                label=trial_label,
            )
        except Exception as exc:
            _finish_inline_status_line()
            print(f"  [WARN] Skipping sequence {i}: {exc}")
            continue
        for k in keys:
            totals[k] += m[k]
        valid += 1

    if valid == 0:
        return {k: 0.0 for k in keys}

    return {k: totals[k] / valid for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective — called once per trial
# ─────────────────────────────────────────────────────────────────────────────
def _run_trial(trial: optuna.Trial,
               loaded_models: list,
               sequences: list,
               lstm_device: torch.device,
               total_trials: int,
               sim_start_time: float) -> float:
    """Suggest parameters, evaluate, return negative composite score."""
    conf_thresh      = trial.suggest_float("conf_thresh",      0.30, 0.85)
    fusion_iou       = trial.suggest_float("fusion_iou",       0.25, 0.75)
    min_support      = trial.suggest_int(  "min_support",      1,    6)
    known_conf       = trial.suggest_float("known_conf",       0.20, 0.80)
    score_margin     = trial.suggest_float("score_margin",     0.05, 0.60)
    disagreement     = trial.suggest_float("disagreement",     0.20, 0.90)
    max_lost         = trial.suggest_int(  "max_lost",         1,    30)
    min_hits         = trial.suggest_int(  "min_hits",         1,    5)
    iou_thresh_track = trial.suggest_float("iou_thresh_track", 0.10, 0.60)
    seq_len          = trial.suggest_int(  "seq_len",          3,    20)
    kalman_r         = trial.suggest_float("kalman_r",         0.5,  20.0)
    kalman_q_vel     = trial.suggest_float("kalman_q_vel",     1e-4, 0.50, log=True)
    kalman_p_vel     = trial.suggest_float("kalman_p_vel",     10.0, 5000.0, log=True)
    w_yolo8n         = trial.suggest_float("w_yolo8n",         0.5,  2.0)
    w_yolo9t         = trial.suggest_float("w_yolo9t",         0.5,  2.0)
    w_yolo10n        = trial.suggest_float("w_yolo10n",        0.5,  2.0)

    model_weights = {
        "yolo8n":  {"bird": w_yolo8n,  "drone": w_yolo8n,  "unknown": w_yolo8n},
        "yolo9t":  {"bird": w_yolo9t,  "drone": w_yolo9t,  "unknown": w_yolo9t},
        "yolo10n": {"bird": w_yolo10n, "drone": w_yolo10n, "unknown": w_yolo10n},
    }

    _patch_globals({
        "CONF_THRESH":               conf_thresh,
        "FUSION_IOU_THRESH":         fusion_iou,
        "MIN_MODEL_SUPPORT":         min_support,
        "KNOWN_FUSED_CONF_THRESH":   known_conf,
        "SCORE_MARGIN_THRESH":       score_margin,
        "DISAGREEMENT_RATIO_THRESH": disagreement,
        "MAX_LOST":                  max_lost,
        "MIN_HITS":                  min_hits,
        "IOU_THRESH_TRACK":          iou_thresh_track,
        "SEQ_LEN":                   seq_len,
        "MODEL_WEIGHTS":             model_weights,
    })
    inf.KalmanBoxTracker.__init__ = _make_kalman_init(kalman_r, kalman_q_vel, kalman_p_vel)

    trial_label = f"Trial {trial.number + 1}/{total_trials}:"
    metrics = evaluate_all(
        loaded_models,
        sequences,
        lstm_device,
        sim_start_time=sim_start_time,
        trial_label=trial_label,
    )

    trial.set_user_attr("auc",      metrics["auc"])
    trial.set_user_attr("sr50",     metrics["sr50"])
    trial.set_user_attr("prec20",   metrics["prec20"])
    trial.set_user_attr("coverage", metrics["coverage"])

    return -metrics["score"]  # minimise negative score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bayesian optimization for WBF+Kalman tracker")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--seed",   type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not VAL_VIDEOS_DIR.is_dir():
        raise FileNotFoundError(f"Validation videos dir not found: {VAL_VIDEOS_DIR}")

    print("=" * 70)
    print("  WBF + Kalman Tracking — Bayesian Optimization Tuner (Optuna TPE)")
    print("=" * 70)
    print(f"  Val dir : {VAL_VIDEOS_DIR}")
    print(f"  Trials  : {args.trials}")
    print(f"  Seed    : {args.seed}")
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

    lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {lstm_device}")

    # Keep Optuna logs quiet so progress output stays readable.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create / resume SQLite-backed study.
    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    storage    = optuna.storages.RDBStorage(f"sqlite:///{OPTUNA_DB}")
    study_name = "wbf_tracking_bayesian_optimization"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"\n  [RESUME] Found existing study with {len(study.trials)} completed trial(s).")
        if study.best_trial is not None:
            print(f"           Best score so far: {-study.best_trial.value:.4f}")
        start_trial = len(study.trials) + 1
    except KeyError:
        sampler = TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction="minimize",
        )
        print("\n  [NEW] Starting fresh optimization study")
        start_trial = 1

    if len(study.trials) >= args.trials:
        print(f"\n  All {args.trials} trials already completed. Delete {OPTUNA_DB} to restart.")
        if study.best_trial is not None:
            print(f"  Best score: {-study.best_trial.value:.4f}")
        return

    print(f"\nStarting optimization ({args.trials} trials, resuming from trial {start_trial})...\n")

    red   = "\033[38;2;255;42;0m"
    reset = "\033[0m"

    sim_start_time = time.time()

    # Seed best score so only genuinely new bests are flagged on resume.
    prior_best_score = -study.best_trial.value if study.best_trial is not None else -float("inf")
    if study.best_trial is not None:
        print(f"  [RESUME] Prior best score: {prior_best_score:.4f}")

    def save_best_json(study_obj: optuna.Study) -> None:
        """Write only the best params to JSON (no history)."""
        if study_obj.best_trial is None:
            return
        bt    = study_obj.best_trial
        bp    = bt.params
        score = -bt.value
        model_weights = {
            "yolo8n":  {"bird": bp["w_yolo8n"],  "drone": bp["w_yolo8n"],  "unknown": bp["w_yolo8n"]},
            "yolo9t":  {"bird": bp["w_yolo9t"],  "drone": bp["w_yolo9t"],  "unknown": bp["w_yolo9t"]},
            "yolo10n": {"bird": bp["w_yolo10n"], "drone": bp["w_yolo10n"], "unknown": bp["w_yolo10n"]},
        }
        output = {
            "score":   score,
            "metrics": {
                "auc":      bt.user_attrs.get("auc",      0.0),
                "sr50":     bt.user_attrs.get("sr50",     0.0),
                "prec20":   bt.user_attrs.get("prec20",   0.0),
                "coverage": bt.user_attrs.get("coverage", 0.0),
            },
            "params": {
                "CONF_THRESH":               bp["conf_thresh"],
                "FUSION_IOU_THRESH":         bp["fusion_iou"],
                "MIN_MODEL_SUPPORT":         bp["min_support"],
                "KNOWN_FUSED_CONF_THRESH":   bp["known_conf"],
                "SCORE_MARGIN_THRESH":       bp["score_margin"],
                "DISAGREEMENT_RATIO_THRESH": bp["disagreement"],
                "MAX_LOST":                  bp["max_lost"],
                "MIN_HITS":                  bp["min_hits"],
                "IOU_THRESH_TRACK":          bp["iou_thresh_track"],
                "SEQ_LEN":                   bp["seq_len"],
                "kalman_R":                  bp["kalman_r"],
                "kalman_Q_vel":              bp["kalman_q_vel"],
                "kalman_P_vel":              bp["kalman_p_vel"],
                "MODEL_WEIGHTS":             model_weights,
            },
        }
        BEST_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_JSON, "w") as f:
            json.dump(output, f, indent=2)

    def on_trial_complete(study_obj: optuna.Study, frozen: optuna.trial.FrozenTrial) -> None:
        score   = -frozen.value
        is_best = (
            study_obj.best_trial is not None
            and study_obj.best_trial.number == frozen.number
            and score > prior_best_score
        )
        auc    = frozen.user_attrs.get("auc",    0.0)
        sr50   = frozen.user_attrs.get("sr50",   0.0)
        prec20 = frozen.user_attrs.get("prec20", 0.0)
        if is_best:
            suffix = (
                f" | AUC: {auc:.4f} | SR@IoU>=0.5: {sr50:.4f}"
                f" | Prec@20px: {prec20:.4f} | Score: {red}{score:.4f}{reset}"
                f" | {red}New best ★{reset}"
            )
        else:
            suffix = (
                f" | AUC: {auc:.4f} | SR@IoU>=0.5: {sr50:.4f}"
                f" | Prec@20px: {prec20:.4f} | Score: {score:.4f}"
            )
        sys.stdout.write(suffix + "\n")
        sys.stdout.flush()
        save_best_json(study_obj)

    def objective(trial: optuna.Trial) -> float:
        return _run_trial(trial, loaded_models, sequences, lstm_device, args.trials, sim_start_time)

    study.optimize(
        objective,
        n_trials=args.trials - len(study.trials),
        show_progress_bar=False,
        callbacks=[on_trial_complete],
    )

    # ── Final summary ──────────────────────────────────────────────────────
    best_trial = study.best_trial
    if best_trial is None:
        raise RuntimeError("No trials completed.")

    bp    = best_trial.params
    score = -best_trial.value

    print(f"\n{'=' * 70}")
    print("  Best trial")
    print(f"{'=' * 70}")
    print(f"  Score      : {score:.4f}")
    print(f"  AUC        : {best_trial.user_attrs.get('auc', 0.0):.4f}")
    print(f"  SR@IoU>=0.5: {best_trial.user_attrs.get('sr50', 0.0):.4f}")
    print(f"  Prec@20px  : {best_trial.user_attrs.get('prec20', 0.0):.4f}")
    print(f"  Coverage   : {best_trial.user_attrs.get('coverage', 0.0):.4f}")
    print("\n  Parameters:")
    for k, v in bp.items():
        print(f"    {k:<28} {v}")

    save_best_json(study)
    print(f"\n  Best params saved → {BEST_JSON}")
    print(f"  Optuna study saved → {OPTUNA_DB}")


if __name__ == "__main__":
    main()
