"""
Bayesian optimization hyperparameter tuning for individual YOLO confidence thresholds.

For each discovered YOLO model checkpoint, this script runs Optuna TPE to tune only
confidence threshold and maximize macro-averaged F1 on validation images.

Features:
  - Per-model SQLite study resume
  - Per-model best JSON snapshot updated after every completed trial
  - Aggregate best JSON summary across all tuned models
  - Inline progress output similar to tune_wbf_6.py

Usage:
    python "tools/tune_yolo_f1.py"
    python "tools/tune_yolo_f1.py" --trials 100 --seed 42
    python "tools/tune_yolo_f1.py" --model yolo9t --trials 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "dataset" / "validation" / "images"
LABELS_DIR = PROJECT_ROOT / "dataset" / "validation" / "labels"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
OUTPUT_DIR = RUNS_DIR / "tune_yolo_f1"
AGGREGATE_BEST_JSON = OUTPUT_DIR / "best_yolo_thresholds.json"

CLASS_NAMES = ["bird", "drone", "unknown"]
IOU_THRESH = 0.5
IMGSZ = 640

IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]

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
    """Return best trial or None when a study has no completed trials yet."""
    try:
        return study_obj.best_trial
    except ValueError:
        return None


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def load_label_file(label_path: Path) -> tuple[list, list]:
    boxes = []
    categories = []
    if not label_path.exists():
        return boxes, categories

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            if cls == 0:
                category = 0
            elif cls == 1:
                category = 1
            else:
                category = 2
            categories.append(category)
            boxes.append((x_center, y_center, w, h))

    return boxes, categories


def xywhn_to_xyxy(box, img_width: int, img_height: int) -> list[float]:
    x_center, y_center, w, h = box
    x1 = (x_center - w / 2.0) * img_width
    y1 = (y_center - h / 2.0) * img_height
    x2 = (x_center + w / 2.0) * img_width
    y2 = (y_center + h / 2.0) * img_height
    return [x1, y1, x2, y2]


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_predictions(gt_boxes: list, pred_boxes: list, iou_thresh: float) -> tuple[dict, set]:
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_gt == 0:
        return {}, set()

    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i in range(n_gt):
        for j in range(n_pred):
            ious[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])

    assignments = {}
    used_pred = set()

    while True:
        if ious.size == 0:
            break
        max_idx = np.unravel_index(np.argmax(ious), ious.shape)
        max_iou = ious[max_idx]
        if max_iou < iou_thresh:
            break

        gt_idx, pred_idx = max_idx
        assignments[gt_idx] = pred_idx
        used_pred.add(pred_idx)
        ious[gt_idx, :] = -1.0
        ious[:, pred_idx] = -1.0

    return assignments, used_pred


def build_confusion_matrix(results: list, label_paths: list[Path], images_dir: Path, iou_thresh: float) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)

    for image_idx, label_path in enumerate(label_paths):
        gt_boxes_xywh, gt_labels = load_label_file(label_path)

        image_name = label_path.stem
        image_path = images_dir / f"{image_name}.jpg"
        if not image_path.exists():
            image_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = images_dir / f"{image_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        if image_path is None:
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        gt_boxes = [xywhn_to_xyxy(box, width, height) for box in gt_boxes_xywh]

        result = results[image_idx]
        pred_boxes = []
        pred_labels = []

        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                cls_id = int(cls)
                pred_labels.append(cls_id if cls_id in (0, 1) else 2)
                pred_boxes.append(list(box))

        assignments, _ = match_predictions(gt_boxes, pred_boxes, iou_thresh)

        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx in assignments:
                pred_idx = assignments[gt_idx]
                pred_label = pred_labels[pred_idx]
                matrix[pred_label, gt_label] += 1
            else:
                matrix[2, gt_label] += 1

    return matrix


def compute_metrics_from_confusion(matrix: np.ndarray) -> tuple[list[dict], dict]:
    """Rows are predicted classes; columns are ground-truth classes."""
    n_classes = matrix.shape[0]
    total = int(matrix.sum())

    per_class_metrics = []
    for c in range(n_classes):
        tp = int(matrix[c, c])
        fp = int(matrix[c, :].sum() - tp)
        fn = int(matrix[:, c].sum() - tp)
        tn = int(total - tp - fp - fn)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = (
            safe_div(2 * precision * recall, precision + recall)
            if not (np.isnan(precision) or np.isnan(recall))
            else float("nan")
        )

        per_class_metrics.append(
            {
                "class": CLASS_NAMES[c],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
            }
        )

    macro_metrics = {
        "Precision": float(np.nanmean([m["Precision"] for m in per_class_metrics])),
        "Recall": float(np.nanmean([m["Recall"] for m in per_class_metrics])),
        "F1-score": float(np.nanmean([m["F1-score"] for m in per_class_metrics])),
    }

    return per_class_metrics, macro_metrics


def find_validation_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[Path], list[Path]]:
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not label_paths:
        raise ValueError(f"No label files found in {labels_dir}")

    image_paths = []
    valid_label_paths = []

    for label_path in label_paths:
        image_name = label_path.stem
        found_image = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{image_name}{ext}"
            if candidate.exists():
                found_image = candidate
                break
        if found_image is None:
            continue

        image_paths.append(found_image)
        valid_label_paths.append(label_path)

    if not image_paths:
        raise ValueError(f"No validation images found in {images_dir}")

    return image_paths, valid_label_paths


def discover_models(runs_dir: Path) -> list[tuple[str, Path]]:
    models = []
    for best_path in sorted(runs_dir.glob("*/weights/best.pt")):
        model_name = best_path.parent.parent.name
        models.append((model_name, best_path))
    return models


def compute_f1_score(
    model,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    conf_thresh: float,
    label: str,
    start_time: float,
) -> tuple[float, float, float, float]:
    """Run model inference on validation images and return objective + metrics."""
    results = []
    total_files = len(image_paths)

    for idx, image_path in enumerate(image_paths, 1):
        result = model.predict(
            source=str(image_path),
            conf=conf_thresh,
            imgsz=IMGSZ,
            verbose=False,
        )
        results.append(result[0] if isinstance(result, list) else result)

        pct = idx / total_files * 100.0
        elapsed_total = time.time() - start_time
        t_hour, t_rem = divmod(int(elapsed_total), 3600)
        t_min, t_sec = divmod(t_rem, 60)
        elapsed_str = f"{t_hour}:{t_min:02d}:{t_sec:02d}"
        status = (
            f"{label} {idx}/{total_files} ({pct:.1f}%) | "
            f"Conf: {conf_thresh:.4f} | Elapsed: {elapsed_str}"
        )
        _print_inline_status(status)

    matrix = build_confusion_matrix(results, valid_label_paths, IMAGES_DIR, IOU_THRESH)
    _per_class, macro_metrics = compute_metrics_from_confusion(matrix)

    precision = float(macro_metrics.get("Precision", np.nan))
    recall = float(macro_metrics.get("Recall", np.nan))
    f1 = float(macro_metrics.get("F1-score", np.nan))

    if np.isnan(precision):
        precision = 0.0
    if np.isnan(recall):
        recall = 0.0
    if np.isnan(f1):
        f1 = 0.0

    f1_neg = -f1
    return f1_neg, precision, recall, f1


class OptimizationTracker:
    """Tracks optimization progress and best score for one model."""

    def __init__(self, start_time: float):
        self.start_time = start_time
        self.best_f1 = -np.inf

    def update(self, precision_score: float, recall_score: float, f1_score: float) -> None:
        is_new_best = f1_score > self.best_f1
        if is_new_best:
            self.best_f1 = f1_score

        red = "\033[38;2;255;42;0m"
        reset = "\033[0m"
        if is_new_best:
            suffix = (
                f" | Precision: {precision_score:.4f} | Recall: {recall_score:.4f} "
                f"| F1: {red}{f1_score:.4f}{reset} | {red}New best *{reset}"
            )
        else:
            suffix = (
                f" | Precision: {precision_score:.4f} | Recall: {recall_score:.4f} "
                f"| F1: {f1_score:.4f}"
            )

        sys.stdout.write(suffix + "\n")
        sys.stdout.flush()

    def finalize(self) -> None:
        _finish_inline_status_line()


def save_best_json_snapshot(
    study_obj: optuna.Study,
    best_json_path: Path,
    model_name: str,
    model_path: Path,
    db_path: Path,
    total_trials_requested: int,
) -> None:
    """Persist current best parameters so progress is visible on disk during runs."""
    best_trial = _safe_best_trial(study_obj)
    if best_trial is None:
        return

    bt = best_trial
    best_conf = float(bt.params["conf_thresh"])
    best_f1 = float(-bt.value)
    best_precision = float(bt.user_attrs.get("precision", 0.0))
    best_recall = float(bt.user_attrs.get("recall", 0.0))

    payload = {
        "model": model_name,
        "model_path": str(model_path),
        "optuna_db": str(db_path),
        "completed_trials": len(study_obj.trials),
        "requested_trials": int(total_trials_requested),
        "best": {
            "f1": best_f1,
            "precision": best_precision,
            "recall": best_recall,
            "conf_thresh": best_conf,
            "trial_number": bt.number,
        },
    }

    best_json_path.parent.mkdir(parents=True, exist_ok=True)
    with best_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def tune_single_model(
    model_name: str,
    model_path: Path,
    image_paths: list[Path],
    valid_label_paths: list[Path],
    trials: int,
    seed: int,
    min_conf: float,
    max_conf: float,
) -> dict | None:
    if not model_path.exists():
        print(f"  [SKIP] {model_name}: checkpoint not found at {model_path}")
        return None

    print(f"\n{'=' * 70}")
    print(f"  Tuning model: {model_name}")
    print(f"  Checkpoint : {model_path}")
    print(f"{'=' * 70}")

    db_path = OUTPUT_DIR / f"optuna_{model_name}.db"
    best_json_path = OUTPUT_DIR / f"best_{model_name}.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
    study_name = f"yolo_threshold_{model_name}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)

        # Mark any trials left in RUNNING state (interrupted mid-trial) as FAILED
        # so they are retried rather than silently skipped on resume.
        running_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.RUNNING
        ]
        if running_trials:
            print(f"  [RESUME] Found {len(running_trials)} interrupted trial(s) — marking as FAILED so they are retried.")
            for t in running_trials:
                storage.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"  [RESUME] Found existing study with {len(completed_trials)} completed trial(s).")
        best_trial = _safe_best_trial(study)
        if best_trial is not None:
            print(f"           Best F1 score so far: {-best_trial.value:.4f}")
        start_trial = len(completed_trials) + 1
    except KeyError:
        # Derive a unique seed per model so each model explores a different
        # initial random sequence even when the same base seed is used.
        model_seed = (seed + hash(model_name)) % (2**31)
        # n_startup_trials controls how many random trials run before the TPE
        # probabilistic model takes over.  Default is 25, which wastes half
        # the budget when --trials=50.  5 random points are enough to seed
        # the model for a single 1-D parameter like conf_thresh.
        sampler = TPESampler(seed=model_seed, n_startup_trials=5)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction="minimize",
        )
        print(f"  [NEW] Starting fresh optimization study (sampler seed: {model_seed})")
        start_trial = 1

    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    if n_complete >= trials:
        print(f"  All {trials} trials already completed. Delete {db_path} to restart.")
        best_trial = _safe_best_trial(study)
        if best_trial is None:
            print("  [WARN] Study has no completed best trial.")
            return None

        save_best_json_snapshot(study, best_json_path, model_name, model_path, db_path, trials)
        bt = best_trial
        return {
            "model": model_name,
            "model_path": str(model_path),
            "optuna_db": str(db_path),
            "best_json": str(best_json_path),
            "completed_trials": n_complete,
            "requested_trials": int(trials),
            "best": {
                "f1": float(-bt.value),
                "precision": float(bt.user_attrs.get("precision", 0.0)),
                "recall": float(bt.user_attrs.get("recall", 0.0)),
                "conf_thresh": float(bt.params["conf_thresh"]),
                "trial_number": int(bt.number),
            },
        }

    print(f"\nStarting optimization ({trials} trials, resuming from trial {n_complete + 1})...\n")

    model = YOLO(str(model_path))
    tracker = OptimizationTracker(start_time=time.time())

    best_trial = _safe_best_trial(study)
    if best_trial is not None:
        prior_best_f1 = -best_trial.value
        tracker.best_f1 = prior_best_f1
        print(f"  [RESUME] Seeding tracker with prior best F1: {prior_best_f1:.4f}")

    def on_trial_complete(study_obj: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        save_best_json_snapshot(study_obj, best_json_path, model_name, model_path, db_path, trials)

    def objective(trial: optuna.Trial) -> float:
        conf_thresh = trial.suggest_float("conf_thresh", min_conf, max_conf)
        n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        trial_label = f"Trial {n_done + 1}/{trials}:"

        f1_neg, precision_score, recall_score, f1_score = compute_f1_score(
            model=model,
            image_paths=image_paths,
            valid_label_paths=valid_label_paths,
            conf_thresh=conf_thresh,
            label=trial_label,
            start_time=tracker.start_time,
        )
        tracker.update(precision_score, recall_score, f1_score)

        trial.set_user_attr("precision", precision_score)
        trial.set_user_attr("recall", recall_score)

        return f1_neg

    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    study.optimize(
        objective,
        n_trials=trials - n_complete,
        show_progress_bar=False,
        callbacks=[on_trial_complete],
    )

    tracker.finalize()

    best_trial = _safe_best_trial(study)
    if best_trial is None:
        print("  [WARN] No trials completed.")
        return None

    save_best_json_snapshot(study, best_json_path, model_name, model_path, db_path, trials)

    n_complete_final = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    bt = best_trial
    result = {
        "model": model_name,
        "model_path": str(model_path),
        "optuna_db": str(db_path),
        "best_json": str(best_json_path),
        "completed_trials": n_complete_final,
        "requested_trials": int(trials),
        "best": {
            "f1": float(-bt.value),
            "precision": float(bt.user_attrs.get("precision", 0.0)),
            "recall": float(bt.user_attrs.get("recall", 0.0)),
            "conf_thresh": float(bt.params["conf_thresh"]),
            "trial_number": int(bt.number),
        },
    }

    print("\n  Best trial summary")
    print(f"  F1        : {result['best']['f1']:.4f}")
    print(f"  Precision : {result['best']['precision']:.4f}")
    print(f"  Recall    : {result['best']['recall']:.4f}")
    print(f"  Conf      : {result['best']['conf_thresh']:.4f}")
    print(f"  Best JSON : {best_json_path}")
    print(f"  Study DB  : {db_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimization for per-model YOLO confidence threshold")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="yolo9t", help="Only tune one model name (run folder name)")
    parser.add_argument("--min-conf", type=float, default=0.55, help="Lower bound for confidence threshold")
    parser.add_argument("--max-conf", type=float, default=0.75, help="Upper bound for confidence threshold")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.min_conf < 0.0 or args.max_conf > 1.0 or args.min_conf >= args.max_conf:
        raise ValueError("Invalid confidence range. Require 0 <= min-conf < max-conf <= 1")

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")
    if not LABELS_DIR.is_dir():
        raise FileNotFoundError(f"Labels dir not found: {LABELS_DIR}")
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(f"Runs dir not found: {RUNS_DIR}")

    print("=" * 70)
    print("  YOLO Per-Model Threshold Tuner (Optuna TPE)")
    print("=" * 70)
    print(f"  Images dir : {IMAGES_DIR}")
    print(f"  Labels dir : {LABELS_DIR}")
    print(f"  Runs dir   : {RUNS_DIR}")
    print(f"  Trials     : {args.trials}")
    print(f"  Seed       : {args.seed}")
    print(f"  Conf range : [{args.min_conf:.2f}, {args.max_conf:.2f}]")

    print("\nLoading validation images and labels...")
    image_paths, valid_label_paths = find_validation_pairs(IMAGES_DIR, LABELS_DIR)
    print(f"  {len(image_paths)} image-label pairs loaded")

    print("\nDiscovering model checkpoints...")
    discovered_models = discover_models(RUNS_DIR)
    if args.model:
        discovered_models = [m for m in discovered_models if m[0] == args.model]

    if not discovered_models:
        if args.model:
            raise RuntimeError(f"No model checkpoint found for model={args.model}")
        raise RuntimeError("No model checkpoints found at runs/detect/*/weights/best.pt")

    print(f"  {len(discovered_models)} model(s) discovered")
    for model_name, model_path in discovered_models:
        print(f"    - {model_name}: {model_path}")

    all_results = {}
    for model_name, model_path in discovered_models:
        try:
            result = tune_single_model(
                model_name=model_name,
                model_path=model_path,
                image_paths=image_paths,
                valid_label_paths=valid_label_paths,
                trials=args.trials,
                seed=args.seed,
                min_conf=args.min_conf,
                max_conf=args.max_conf,
            )
            if result is not None:
                all_results[model_name] = result
        except Exception as exc:
            _finish_inline_status_line()
            print(f"\n  [WARN] Failed tuning {model_name}: {exc}")
            continue

    if not all_results:
        raise RuntimeError("No model produced a tuning result.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "generated_at_epoch": int(time.time()),
        "settings": {
            "trials_per_model": int(args.trials),
            "seed": int(args.seed),
            "min_conf": float(args.min_conf),
            "max_conf": float(args.max_conf),
            "iou_thresh": float(IOU_THRESH),
            "imgsz": int(IMGSZ),
            "images_dir": str(IMAGES_DIR),
            "labels_dir": str(LABELS_DIR),
        },
        "models": all_results,
    }

    with AGGREGATE_BEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nAggregate best summary saved: {AGGREGATE_BEST_JSON}")

    ranking = sorted(
        all_results.items(),
        key=lambda kv: kv[1]["best"]["f1"],
        reverse=True,
    )

    print(f"\n{'=' * 70}")
    print("  Best F1 per model")
    print(f"{'=' * 70}")
    for idx, (model_name, result) in enumerate(ranking, 1):
        best = result["best"]
        print(
            f"  {idx:>2}. {model_name:<15} "
            f"F1={best['f1']:.4f} "
            f"Prec={best['precision']:.4f} "
            f"Rec={best['recall']:.4f} "
            f"Conf={best['conf_thresh']:.4f}"
        )


if __name__ == "__main__":
    main()
