"""
Bayesian optimization hyperparameter tuning for the WBF + Kalman tracking pipeline.

Loads ensemble models once, then uses Optuna's TPE sampler to tune:
  - CONF_THRESH, FUSION_IOU_THRESH
    - Per-model confidence thresholds (conf_yolo8n, conf_yolo11n, conf_yolo26n)
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
INFERENCE_PATH = PROJECT_ROOT / "scripts" / "model_testing" / "inference_video_wbf_tracking.py"
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

# One-time cache inference threshold. Trials later apply CONF_THRESH filtering
# on cached detections, so this must stay very low.
CONF_INFER = 0.001
MODEL_CONF_RANGE = (0.30, 0.85)
TARGET_MODEL_NAMES = ("yolo8n", "yolo11n", "yolo26n")

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


def _safe_best_trial(study_obj: optuna.Study):
    """Return best trial or None when no completed trial exists yet."""
    try:
        return study_obj.best_trial
    except ValueError:
        return None


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
    candidate_paths = {
        model_name: model_path for model_name, model_path in inf.MODELS
    }
    for model_name in TARGET_MODEL_NAMES:
        model_path = candidate_paths.get(
            model_name,
            PROJECT_ROOT / "runs" / "detect" / model_name / "weights" / "best.pt",
        )
        if not model_path.exists():
            print(f"  [SKIP] {model_name}: not found at {model_path}")
            continue
        from ultralytics import YOLO  # noqa: PLC0415
        model     = YOLO(str(model_path))
        class_map = inf.build_model_class_map(model)
        loaded.append((model_name, model, class_map))
        print(f"  [OK]   {model_name}")
    if not loaded:
        raise RuntimeError("No target ensemble models loaded. Check model paths.")
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


def _fuse_from_cached_raw(frame_raw_dets: list[dict]) -> list[dict]:
    """Build fused detections from cached model detections for one frame."""
    all_detections: list[dict] = []
    model_conf_thresh = getattr(inf, "MODEL_CONF_THRESH", {})
    global_conf_thresh = float(getattr(inf, "CONF_THRESH", 0.0))

    for det in frame_raw_dets:
        per_model_thresh = float(model_conf_thresh.get(det["model"], global_conf_thresh))
        effective_thresh = max(global_conf_thresh, per_model_thresh)
        if det["confidence"] < effective_thresh:
            continue
        cls_name = inf.class_name_from_id(int(det["class_id"]))
        mw = inf.MODEL_WEIGHTS.get(det["model"], {}).get(cls_name, 1.0)
        all_detections.append({
            "box": det["box"],
            "class_id": det["class_id"],
            "confidence": det["confidence"],
            "model": det["model"],
            "weighted_score": float(mw * det["confidence"]),
        })

    if not all_detections:
        return []

    fused: list[dict] = []
    for cluster in inf.cluster_detections(all_detections, inf.FUSION_IOU_THRESH):
        item = inf.fuse_cluster(cluster)
        if item is not None:
            fused.append(item)
    return fused


def cache_sequence_detections(
    loaded_models: list,
    sequence_specs: list[tuple[Path, list, list]],
    sim_start_time: float,
) -> list[dict]:
    """Run ensemble inference once over all frames and cache raw detections."""
    cached_sequences: list[dict] = []
    n_seq = len(sequence_specs)

    for seq_idx, (video_path, gt_exist, gt_rect) in enumerate(sequence_specs, 1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for caching: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_raw: list[list[dict]] = []
        processed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_raw: list[dict] = []
            for model_name, model, class_map in loaded_models:
                predict_kwargs = {
                    "source": frame,
                    "conf": CONF_INFER,
                    "imgsz": inf.IMG_SIZE,
                    "save": False,
                    "show": False,
                    "verbose": False,
                }
                if inf.DEVICE:
                    predict_kwargs["device"] = inf.DEVICE

                results = model.predict(**predict_kwargs)
                r = results[0] if isinstance(results, list) else results

                if not hasattr(r, "boxes") or len(r.boxes) == 0:
                    continue

                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                confidences = r.boxes.conf.cpu().numpy().astype(float)

                for box, cls_id, conf in zip(boxes_xyxy, class_ids, confidences):
                    cls_name = class_map.get(int(cls_id), inf.class_name_from_id(int(cls_id)))
                    cls_idx = 0 if cls_name == "bird" else 1 if cls_name == "drone" else 2
                    frame_raw.append({
                        "box": [float(v) for v in box.tolist()],
                        "class_id": int(cls_idx),
                        "confidence": float(conf),
                        "model": model_name,
                    })

            frames_raw.append(frame_raw)
            processed += 1

            elapsed_total = time.time() - sim_start_time
            t_hour, t_rem = divmod(int(elapsed_total), 3600)
            t_min, t_sec = divmod(t_rem, 60)
            elapsed_str = f"{t_hour}:{t_min:02d}:{t_sec:02d}"
            pct = (processed / total_frames * 100.0) if total_frames > 0 else 0.0
            _print_inline_status(
                f"  Caching Video {seq_idx}/{n_seq} | Frame {processed}/{total_frames} ({pct:.1f}%)"
                f" | Elapsed: {elapsed_str}"
            )

        cap.release()
        cached_sequences.append(
            {
                "video_path": video_path,
                "gt_exist": gt_exist,
                "gt_rect": gt_rect,
                "width": width,
                "height": height,
                "frames_raw": frames_raw,
            }
        )

    _finish_inline_status_line()
    return cached_sequences


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop (no video writing)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(seq_cache: dict,
             lstm_device: torch.device, seq_idx: int | None = None,
             total_seq: int | None = None,
             sim_start_time: float | None = None,
             label: str = "") -> dict:
    """Run the full pipeline on one cached video and return evaluation metrics."""

    # Reset track ID counter so each trial starts fresh
    inf.KalmanBoxTracker._next_id = 0

    tracker = inf.MultiObjectTracker(lstm_device=lstm_device)

    gt_exist = seq_cache["gt_exist"]
    gt_rect = seq_cache["gt_rect"]
    width = int(seq_cache["width"])
    height = int(seq_cache["height"])
    frames_raw = seq_cache["frames_raw"]
    total_frames = len(frames_raw)

    frame_ious:     list[float] = []
    frame_dists:    list[float] = []
    covered_frames: int = 0
    exist1_frames:  int = 0
    processed:      int = 0

    for frame_raw in frames_raw:
        fused      = _fuse_from_cached_raw(frame_raw)
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
        status = f"    {trial_prefix}{video_prefix}Elapsed: {elapsed_str}"
        _print_inline_status(status)

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
def evaluate_all(cached_sequences: list[dict],
                 lstm_device: torch.device,
                 sim_start_time: float | None = None,
                 trial_label: str = "") -> dict:
    """Evaluate across all sequences and return mean metrics."""
    keys = ("auc", "sr50", "prec20", "coverage", "score")
    totals = {k: 0.0 for k in keys}
    valid  = 0

    n_seq = len(cached_sequences)
    for i, seq_cache in enumerate(cached_sequences, 1):
        try:
            m = evaluate(
                seq_cache,
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
               cached_sequences: list[dict],
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
    conf_yolo8n      = trial.suggest_float("conf_yolo8n",      *MODEL_CONF_RANGE)
    conf_yolo11n     = trial.suggest_float("conf_yolo11n",     *MODEL_CONF_RANGE)
    conf_yolo26n     = trial.suggest_float("conf_yolo26n",     *MODEL_CONF_RANGE)
    w_yolo8n         = trial.suggest_float("w_yolo8n",         0.5,  2.0)
    w_yolo11n        = trial.suggest_float("w_yolo11n",        0.5,  2.0)
    w_yolo26n        = trial.suggest_float("w_yolo26n",        0.5,  2.0)

    model_conf_thresh = {
        "yolo8n": float(conf_yolo8n),
        "yolo11n": float(conf_yolo11n),
        "yolo26n": float(conf_yolo26n),
    }

    model_weights = {
        "yolo8n":  {"bird": w_yolo8n,  "drone": w_yolo8n,  "unknown": w_yolo8n},
        "yolo11n": {"bird": w_yolo11n, "drone": w_yolo11n, "unknown": w_yolo11n},
        "yolo26n": {"bird": w_yolo26n, "drone": w_yolo26n, "unknown": w_yolo26n},
    }

    _patch_globals({
        "CONF_THRESH":               conf_thresh,
        "MODEL_CONF_THRESH":         model_conf_thresh,
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
        cached_sequences,
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
    parser.add_argument("--trials", type=int, default=1000, help="Number of optimization trials")
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
    print(f"  JSON out: {BEST_JSON}")
    print(f"  DB out  : {OPTUNA_DB}")
    print()

    print("Discovering validation sequences...")
    sequence_specs: list[tuple[Path, list, list]] = []
    for seq_dir in sorted(VAL_VIDEOS_DIR.iterdir()):
        video_path = seq_dir / "visible.mp4"
        gt_path    = seq_dir / "visible.json"
        if not video_path.exists() or not gt_path.exists():
            print(f"  [SKIP] {seq_dir.name}: missing video or GT")
            continue
        gt_exist, gt_rect = load_gt(gt_path)
        sequence_specs.append((video_path, gt_exist, gt_rect))
    if not sequence_specs:
        raise RuntimeError("No valid validation sequences found.")
    print(f"  {len(sequence_specs)} sequences loaded")

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
        _resume_best = _safe_best_trial(study)
        if _resume_best is not None:
            print(f"           Best score so far: {-_resume_best.value:.4f}")
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
        _done_best = _safe_best_trial(study)
        if _done_best is not None:
            print(f"  Best score: {-_done_best.value:.4f}")
        return

    print("\nCaching detections once (GPU pass) for all validation videos...")
    cache_start_time = time.time()
    cached_sequences = cache_sequence_detections(loaded_models, sequence_specs, cache_start_time)
    print(f"  Cache ready for {len(cached_sequences)} sequence(s). Trials now run CPU-only fusion/tracking.")

    print(f"\nStarting optimization ({args.trials} trials, resuming from trial {start_trial})...\n")

    red   = "\033[38;2;255;42;0m"
    reset = "\033[0m"

    sim_start_time = time.time()

    # Seed best score so only genuinely new bests are flagged on resume.
    _prior_best = _safe_best_trial(study)
    prior_best_score = -_prior_best.value if _prior_best is not None else -float("inf")
    if _prior_best is not None:
        print(f"  [RESUME] Prior best score: {prior_best_score:.4f}")

    def save_best_json(study_obj: optuna.Study) -> None:
        """Write only the best params to JSON (no history)."""
        bt = _safe_best_trial(study_obj)
        if bt is None:
            return
        bp    = bt.params
        score = -bt.value
        model_weights = {
            "yolo8n":  {"bird": bp["w_yolo8n"],  "drone": bp["w_yolo8n"],  "unknown": bp["w_yolo8n"]},
            "yolo11n": {"bird": bp["w_yolo11n"], "drone": bp["w_yolo11n"], "unknown": bp["w_yolo11n"]},
            "yolo26n": {"bird": bp["w_yolo26n"], "drone": bp["w_yolo26n"], "unknown": bp["w_yolo26n"]},
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
                "MODEL_CONF_THRESH": {
                    "yolo8n": bp.get("conf_yolo8n", bp["conf_thresh"]),
                    "yolo11n": bp.get("conf_yolo11n", bp["conf_thresh"]),
                    "yolo26n": bp.get("conf_yolo26n", bp["conf_thresh"]),
                },
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
        best_now = _safe_best_trial(study_obj)
        is_best = (
            best_now is not None
            and best_now.number == frozen.number
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
        return _run_trial(trial, cached_sequences, lstm_device, args.trials, sim_start_time)

    study.optimize(
        objective,
        n_trials=args.trials - len(study.trials),
        show_progress_bar=False,
        callbacks=[on_trial_complete],
    )

    # ── Final summary ──────────────────────────────────────────────────────
    best_trial = _safe_best_trial(study)
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
