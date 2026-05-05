"""
Bayesian optimization hyperparameter tuning for weighted boxes fusion using Optuna.

Loads ensemble models once, then uses Optuna's TPE sampler to tune:
  - MIN_MODEL_SUPPORT
  - KNOWN_FUSED_CONF_THRESH
  - SCORE_MARGIN_THRESH
  - DISAGREEMENT_RATIO_THRESH
  - Per-model per-class weights (informed by per-class metrics)

Optimizes macro-averaged F1 score on validation images.

Usage:
    python "tools/tune_weighted_boxes_fusion.py"
    python "tools/tune_weighted_boxes_fusion.py" --trials 50 --seed 42
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

# ── resolve paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WBF_PATH = PROJECT_ROOT / "scripts" / "algorithms" / "weighted_boxes_fusion.py"
if not WBF_PATH.exists():
    raise FileNotFoundError(f"WBF script not found: {WBF_PATH}")

_spec = importlib.util.spec_from_file_location("weighted_boxes_fusion", WBF_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot create import spec for: {WBF_PATH}")
wbf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wbf)

# ── config ───────────────────────────────────────────────────────────────────
IMAGES_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "validation" / "labels"

BEST_JSON = PROJECT_ROOT / "runs" / "detect" / "tune_wbf" / "best_params_bayesian.json"
OPTUNA_DB = PROJECT_ROOT / "runs" / "detect" / "tune_wbf" / "optuna_study.db"

# ── Tuning search ranges ──────────────────────────────────────────────────────
MIN_SUPPORT_RANGE   = (1, 6)          # (min, max) int
KNOWN_CONF_RANGE    = (0.30, 0.90)    # (min, max) float
SCORE_MARGIN_RANGE  = (0.00, 1.00)    # (min, max) float
DISAGREEMENT_RANGE  = (0.00, 1.00)    # (min, max) float
MODEL_WEIGHT_RANGE  = (0.50, 2.00)    # (min, max) float, per-model per-class

# ── Initial / baseline parameter values (sourced from weighted_boxes_fusion.py) ─
INIT_MIN_SUPPORT   = wbf.MIN_MODEL_SUPPORT
INIT_KNOWN_CONF    = wbf.KNOWN_FUSED_CONF_THRESH
INIT_SCORE_MARGIN  = wbf.SCORE_MARGIN_THRESH
INIT_DISAGREEMENT  = wbf.DISAGREEMENT_RATIO_THRESH

# Model names in order (must match wbf.MODELS)
MODEL_NAMES = [name for name, _ in wbf.MODELS]
CLASS_NAMES = wbf.CLASS_NAMES
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


def _patch_globals(params: dict) -> None:
    """Patch module globals with trial parameters."""
    for key, val in params.items():
        if not key.startswith("w_"):  # Skip individual weight keys
            setattr(wbf, key, val)
        
    # Set MODEL_WEIGHTS from per-class weights
    if "MODEL_WEIGHTS" in params:
        wbf.MODEL_WEIGHTS = params["MODEL_WEIGHTS"]


def _encode_params(min_support: int, known_conf: float, score_margin: float,
                   disagreement: float, weights_list: list) -> dict:
    """Convert optimization space parameters to trial parameters."""
    # weights_list is flattened: [w_model0_class0, w_model0_class1, ..., w_model5_class2]
    model_weights = {}
    idx = 0
    for model_name in MODEL_NAMES:
        model_weights[model_name] = {
            "bird": float(weights_list[idx]),
            "drone": float(weights_list[idx + 1]),
            "unknown": float(weights_list[idx + 2]),
        }
        idx += 3
    
    return {
        "MIN_MODEL_SUPPORT": int(min_support),
        "KNOWN_FUSED_CONF_THRESH": float(known_conf),
        "SCORE_MARGIN_THRESH": float(score_margin),
        "DISAGREEMENT_RATIO_THRESH": float(disagreement),
        "MODEL_WEIGHTS": model_weights,
    }


def compute_f1_score(loaded_models: list, image_paths: list, valid_label_paths: list,
                     label: str = "", start_time: float = None) -> tuple[float, float, float, float]:
    """
    Run inference on all validation images and return macro-averaged F1 score.
    """
    fused_results_per_image = []
    total_files = len(image_paths)
    t0 = start_time if start_time is not None else time.time()

    for idx, image_path in enumerate(image_paths, 1):
        fused, _ = wbf.run_weighted_boxes_fusion_on_image(loaded_models, image_path)
        fused_results_per_image.append(fused)

        pct = idx / total_files * 100.0
        elapsed_total = time.time() - t0
        t_hour, t_rem = divmod(int(elapsed_total), 3600)
        t_min, t_sec = divmod(t_rem, 60)
        elapsed_str = f"{t_hour}:{t_min:02d}:{t_sec:02d}"
        prefix = f"{label} " if label else ""
        status = f"    {prefix}{idx}/{total_files} ({pct:.1f}%) | Elapsed: {elapsed_str}"
        _print_inline_status(status)
    
    # Compute confusion matrix and metrics
    matrix, _, _ = wbf.build_confusion_matrix(
        fused_results_per_image,
        valid_label_paths,
        IMAGES_DIR,
        wbf.IOU_THRESH,
        verbose=False,
    )
    
    _, macro_metrics, _ = wbf.compute_metrics_from_confusion(matrix)
    precision_score = float(macro_metrics.get("Precision", np.nan))
    recall_score = float(macro_metrics.get("Recall", np.nan))
    f1_score = float(macro_metrics.get("F1-score", np.nan))

    if np.isnan(precision_score):
        precision_score = 0.0
    if np.isnan(recall_score):
        recall_score = 0.0
    if np.isnan(f1_score):
        f1_neg = 1.0
        f1_score = 0.0
    else:
        f1_neg = float(-f1_score)

    # Return objective value and metrics for trial reporting.
    return f1_neg, precision_score, recall_score, f1_score


def load_models() -> list:
    """Load ensemble models once."""
    loaded = []
    for model_name, model_path in wbf.MODELS:
        if not model_path.exists():
            print(f"  [SKIP] {model_name}: not found at {model_path}")
            continue
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        class_map = wbf.build_model_class_map(model)
        loaded.append((model_name, model, class_map))
        print(f"  [OK]   {model_name}")
    if not loaded:
        raise RuntimeError("No ensemble models loaded.")
    return loaded


def init_weights_from_baseline(loaded_models: list, image_paths: list,
                               valid_label_paths: list) -> dict:
    """
    Run baseline (all weights = 1.0) and derive initial weights from per-class metrics.
    Weights are scaled by each model's F1 score on that class.
    """
    print("\n  Computing per-model per-class metrics for weight initialization...")
    per_model_results = {model_name: [] for model_name, _, _ in loaded_models}
    total_files = len(image_paths)
    last_status_len = 0
    start_time = time.time()
    
    for idx, image_path in enumerate(image_paths, 1):
        _, per_model_preds = wbf.run_weighted_boxes_fusion_on_image(loaded_models, image_path)
        for model_name in per_model_results:
            per_model_results[model_name].append(per_model_preds[model_name])

        # Update progress for every processed file on a single terminal line.
        pct = (idx / total_files * 100.0) if total_files > 0 else 0.0
        elapsed_total = time.time() - start_time
        t_hour, t_rem = divmod(int(elapsed_total), 3600)
        t_min, t_sec = divmod(t_rem, 60)
        elapsed_str = f"{t_hour}:{t_min:02d}:{t_sec:02d}"
        status = f"    {idx}/{total_files} ({pct:.1f}%) | Elapsed: {elapsed_str}"
        padded = status.ljust(last_status_len)
        print(f"\r{padded}", end="", flush=True)
        last_status_len = len(status)

    print()
    print("    Baseline inference completed.")
    
    # Compute per-model per-class metrics
    weights = {}
    for model_name in MODEL_NAMES:
        model_matrix, _, _ = wbf.build_confusion_matrix(
            per_model_results[model_name],
            valid_label_paths,
            IMAGES_DIR,
            wbf.IOU_THRESH,
            verbose=False,
        )
        per_class_metrics, _, _ = wbf.compute_metrics_from_confusion(model_matrix)
        
        weights[model_name] = {}
        for metric in per_class_metrics:
            cls_name = metric["class"]
            f1 = metric["F1-score"]
            # Scale weight by F1 score, clamped to [0.5, 2.0]
            if np.isnan(f1):
                weight = 1.0
            else:
                weight = np.clip(f1 * 2.0, 0.5, 2.0)  # Range [0.5, 2.0]
            weights[model_name][cls_name] = float(weight)
    
    return weights


class OptimizationTracker:
    """Tracks optimization progress and best score."""
    def __init__(self, start_time: float):
        self.start_time = start_time
        self.best_f1 = -np.inf
        self.best_precision = 0.0
        self.best_recall = 0.0
    
    def update(self, trial_number: int, total_trials: int,
               precision_score: float, recall_score: float, f1_score: float) -> None:
        """Update tracker with new result."""
        is_new_best = f1_score > self.best_f1
        if is_new_best:
            self.best_f1 = f1_score
            self.best_precision = precision_score
            self.best_recall = recall_score
        
        RED = "\033[38;2;255;42;0m"
        RESET = "\033[0m"

        if is_new_best:
            suffix = (
                f" | Precision: {precision_score:.4f} | Recall: {recall_score:.4f} | F1: {RED}{f1_score:.4f}{RESET}"
                f" | {RED}New best ★{RESET}"
            )
        else:
            suffix = f" | Precision: {precision_score:.4f} | Recall: {recall_score:.4f} | F1: {f1_score:.4f}"
        sys.stdout.write(suffix + "\n")
        sys.stdout.flush()

    def finalize(self) -> None:
        """Move to the next line after the final status update."""
        _finish_inline_status_line()


def main():
    parser = argparse.ArgumentParser(description="Bayesian optimization for weighted boxes fusion")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")
    if not LABELS_DIR.is_dir():
        raise FileNotFoundError(f"Labels dir not found: {LABELS_DIR}")
    
    print("=" * 70)
    print("  Weighted Boxes Fusion — Bayesian Optimization Tuner (Optuna TPE)")
    print("=" * 70)
    print(f"  Images dir: {IMAGES_DIR}")
    print(f"  Labels dir: {LABELS_DIR}")
    print(f"  Trials    : {args.trials}")
    print(f"  Seed      : {args.seed}")
    print()
    
    # Load validation images
    print("Loading validation images and labels...")
    label_paths = sorted(list(LABELS_DIR.glob("*.txt")))
    if not label_paths:
        raise ValueError(f"No label files found in {LABELS_DIR}")
    
    image_paths = []
    valid_label_paths = []
    for label_path in label_paths:
        image_name = label_path.stem
        image_path = IMAGES_DIR / f"{image_name}.jpg"
        if not image_path.exists():
            for ext in [".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = IMAGES_DIR / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path.exists():
            image_paths.append(image_path)
            valid_label_paths.append(label_path)
    
    if not image_paths:
        raise ValueError(f"No validation images found in {IMAGES_DIR}")
    print(f"  {len(image_paths)} images loaded")
    
    # Load ensemble models
    print("\nLoading ensemble models...")
    loaded_models = load_models()
    
    # Initialize weights from baseline
    print("\nBuilding optimization space...")
    init_weights = init_weights_from_baseline(loaded_models, image_paths, valid_label_paths)
    
    # Build initial point with defaults
    init_params = {
        "min_support": INIT_MIN_SUPPORT,
        "known_conf":  INIT_KNOWN_CONF,
        "score_margin": INIT_SCORE_MARGIN,
        "disagreement": INIT_DISAGREEMENT,
    }
    
    # Add initial weights from baseline analysis
    for model_name in MODEL_NAMES:
        for class_name in CLASS_NAMES:
            init_params[f"w_{model_name}_{class_name}"] = init_weights[model_name][class_name]
    
    # Evaluate initial point
    opt_start_time = time.time()
    print("Evaluating initial point...")
    x0_weights = [init_weights[model_name][class_name] for model_name in MODEL_NAMES for class_name in CLASS_NAMES]
    params_init = _encode_params(INIT_MIN_SUPPORT, INIT_KNOWN_CONF, INIT_SCORE_MARGIN, INIT_DISAGREEMENT, x0_weights)
    _patch_globals(params_init)
    y0_neg, y0_precision, y0_recall, y0_f1 = compute_f1_score(
        loaded_models,
        image_paths,
        valid_label_paths,
        label="Initial:",
        start_time=opt_start_time,
    )
    _finish_inline_status_line()
    print(f"  Initial Precision: {y0_precision:.4f}")
    print(f"  Initial Recall   : {y0_recall:.4f}")
    print(f"  Initial F1 score : {y0_f1:.4f}")
    
    # Keep Optuna logs quiet so progress output stays readable.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create Optuna storage (SQLite database for persistence)
    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{OPTUNA_DB}")
    
    # Check if study exists
    study_name = "wbf_bayesian_optimization"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"\n  [RESUME] Found existing study with {len(study.trials)} completed trial(s).")
        if study.best_trial is not None:
            print(f"           Best F1 score so far: {-study.best_trial.value:.4f}")
        start_trial = len(study.trials) + 1
    except KeyError:
        # Create new study with TPE sampler
        sampler = TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction="minimize"  # Minimize negative F1 (= maximize F1)
        )
        print("\n  [NEW] Starting fresh optimization study")
        start_trial = 1
    
    # Check if already completed
    if len(study.trials) >= args.trials:
        print(f"\n  All {args.trials} trials already completed. Delete {OPTUNA_DB} to restart.")
        if study.best_trial is not None:
            best_f1 = -study.best_trial.value
            print(f"  Best F1 score: {best_f1:.4f}")
        return
    
    print(f"\nStarting optimization ({args.trials} trials, resuming from trial {start_trial})...\n")
    
    sim_start_time = opt_start_time
    tracker = OptimizationTracker(sim_start_time)

    # Seed tracker with best score already in the study (so resume doesn't re-announce old bests)
    if study.best_trial is not None:
        prior_best_f1 = -study.best_trial.value
        tracker.best_f1 = prior_best_f1
        prior_p, prior_r = 0.0, 0.0
        if BEST_JSON.exists():
            with open(BEST_JSON) as _f:
                _j = json.load(_f)
            prior_p = _j.get("precision", 0.0)
            prior_r = _j.get("recall", 0.0)
        tracker.best_precision = prior_p
        tracker.best_recall = prior_r
        print(f"  [RESUME] Seeding tracker with prior best F1: {prior_best_f1:.4f} (Precision: {prior_p:.4f}, Recall: {prior_r:.4f})")

    def save_best_json_snapshot(study_obj: optuna.Study) -> None:
        """Persist current best parameters so progress is visible on disk during runs."""
        if study_obj.best_trial is None:
            return

        best_trial_obj = study_obj.best_trial
        best_f1_obj = -best_trial_obj.value
        best_params_obj = best_trial_obj.params

        model_weights_obj = {}
        for model_name in MODEL_NAMES:
            model_weights_obj[model_name] = {}
            for class_name in CLASS_NAMES:
                model_weights_obj[model_name][class_name] = best_params_obj.get(
                    f"w_{model_name}_{class_name}",
                    1.0,
                )

        best_precision_obj = best_trial_obj.user_attrs.get("precision", 0.0)
        best_recall_obj = best_trial_obj.user_attrs.get("recall", 0.0)

        output_obj = {
            "score": best_f1_obj,
            "precision": best_precision_obj,
            "recall": best_recall_obj,
            "params": {
                "MIN_MODEL_SUPPORT": best_params_obj["min_support"],
                "KNOWN_FUSED_CONF_THRESH": best_params_obj["known_conf"],
                "SCORE_MARGIN_THRESH": best_params_obj["score_margin"],
                "DISAGREEMENT_RATIO_THRESH": best_params_obj["disagreement"],
                "MODEL_WEIGHTS": model_weights_obj,
            },
        }

        BEST_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_JSON, "w") as f:
            json.dump(output_obj, f, indent=2)

    def on_trial_complete(study_obj: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        save_best_json_snapshot(study_obj)
    
    # Define objective function with closure
    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters
        min_support = trial.suggest_int(  "min_support", *MIN_SUPPORT_RANGE)
        known_conf  = trial.suggest_float("known_conf",   *KNOWN_CONF_RANGE)
        score_margin = trial.suggest_float("score_margin", *SCORE_MARGIN_RANGE)
        disagreement = trial.suggest_float("disagreement", *DISAGREEMENT_RANGE)
        
        # Suggest weights for each model and class
        weights_list = []
        for model_name in MODEL_NAMES:
            for class_name in CLASS_NAMES:
                w = trial.suggest_float(f"w_{model_name}_{class_name}", *MODEL_WEIGHT_RANGE)
                weights_list.append(w)
        
        # Encode and patch parameters
        params = _encode_params(min_support, known_conf, score_margin, disagreement, weights_list)
        _patch_globals(params)
        
        # Compute F1 score (shows per-image progress inline)
        trial_label = f"Trial {trial.number + 1}/{args.trials}:"
        f1_neg, precision_score, recall_score, f1_score = compute_f1_score(
            loaded_models,
            image_paths,
            valid_label_paths,
            label=trial_label,
            start_time=sim_start_time,
        )

        # Update summary line after trial completes
        tracker.update(trial.number + 1, args.trials, precision_score, recall_score, f1_score)

        # Store precision/recall so resume can seed the tracker correctly
        trial.set_user_attr("precision", precision_score)
        trial.set_user_attr("recall", recall_score)

        # Return negative F1 for minimization
        return f1_neg
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=args.trials - len(study.trials),
        show_progress_bar=False,
        callbacks=[on_trial_complete],
    )
    
    # Clear progress line
    tracker.finalize()
    
    # ── Extract best result ───────────────────────────────────────────────────
    best_trial = study.best_trial
    if best_trial is None:
        raise RuntimeError("No trials completed.")
    
    best_f1 = -best_trial.value
    best_params_dict = best_trial.params
    
    # Reconstruct MODEL_WEIGHTS from flat params
    model_weights = {}
    for model_name in MODEL_NAMES:
        model_weights[model_name] = {}
        for class_name in CLASS_NAMES:
            model_weights[model_name][class_name] = best_params_dict.get(f"w_{model_name}_{class_name}", 1.0)
    
    best_params = {
        "MIN_MODEL_SUPPORT": best_params_dict["min_support"],
        "KNOWN_FUSED_CONF_THRESH": best_params_dict["known_conf"],
        "SCORE_MARGIN_THRESH": best_params_dict["score_margin"],
        "DISAGREEMENT_RATIO_THRESH": best_params_dict["disagreement"],
        "MODEL_WEIGHTS": model_weights,
    }
    
    print(f"\n{'=' * 70}")
    print("  Best trial")
    print(f"{'=' * 70}")
    print(f"  F1 Score : {best_f1:.4f}")
    print("\n  Parameters:")
    print(f"    MIN_MODEL_SUPPORT            : {best_params['MIN_MODEL_SUPPORT']}")
    print(f"    KNOWN_FUSED_CONF_THRESH     : {best_params['KNOWN_FUSED_CONF_THRESH']:.4f}")
    print(f"    SCORE_MARGIN_THRESH         : {best_params['SCORE_MARGIN_THRESH']:.4f}")
    print(f"    DISAGREEMENT_RATIO_THRESH   : {best_params['DISAGREEMENT_RATIO_THRESH']:.4f}")
    print("\n  Per-model per-class weights:")
    for model_name in MODEL_NAMES:
        print(f"    {model_name}:")
        for class_name in CLASS_NAMES:
            w = best_params["MODEL_WEIGHTS"][model_name][class_name]
            print(f"      {class_name}: {w:.3f}")
    
    # Save final best snapshot
    save_best_json_snapshot(study)
    print(f"\n  Best params saved → {BEST_JSON}")
    print(f"  Optuna study saved → {OPTUNA_DB}")


if __name__ == "__main__":
    main()
