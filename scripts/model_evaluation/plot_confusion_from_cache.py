"""
Read eval_cache JSON files from runs/detect/weights/<model>/eval_cache/
and runs/fasterrcnn/train/eval_cache/ and generate percentage-normalised
confusion matrix PNGs for each model.
Output is saved to runs/detect/weights/<model_name>.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECT_DIR = PROJECT_ROOT / "runs" / "detect"
WEIGHTS_DIR = DETECT_DIR / "weights"
FASTERRCNN_CACHE_DIR = PROJECT_ROOT / "runs" / "fasterrcnn" / "train" / "eval_cache"

CLASS_NAMES = ["bird", "drone", "unknown"]

MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n"]

# Visual settings — identical to eval_yolo.py
TICK_LABEL_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
CELL_VALUE_FONTSIZE = 33
PREDICTED_LABEL_PAD = -14


def format_run_display_name(run_name: str) -> str:
    lower_name = run_name.lower()
    if lower_name.startswith("yolo") and len(run_name) > 4:
        suffix = run_name[4:]
        if suffix and suffix[0].isdigit():
            return f"YOLOv{suffix}"
    return run_name


def pick_cache_file(model_name: str) -> Path | None:
    """Return the best-matching test cache from runs/detect/<model>/eval_cache/."""
    cache_dir = DETECT_DIR / model_name / "eval_cache"
    if not cache_dir.exists():
        return None

    # Try to match the conf threshold from best_<model>.json, preferring test_ prefix
    best_json = WEIGHTS_DIR / f"best_{model_name}.json"
    if best_json.exists():
        with best_json.open(encoding="utf-8") as f:
            data = json.load(f)
        conf = float(data["best"]["conf_thresh"])
        conf_str = f"{conf:.6f}".replace(".", "p")
        for prefix in ("test_metrics_", "metrics_"):
            for cf in cache_dir.iterdir():
                if cf.name.startswith(prefix) and conf_str in cf.name:
                    return cf

    # Fallback: most recent test_ file, then any file
    test_files = sorted(cache_dir.glob("test_metrics_*"), key=lambda p: p.stat().st_mtime)
    if test_files:
        return test_files[-1]
    all_files = list(cache_dir.iterdir())
    return max(all_files, key=lambda p: p.stat().st_mtime) if all_files else None


def load_matrix(cache_file: Path) -> np.ndarray | None:
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    matrix_data = payload.get("matrix")
    if not isinstance(matrix_data, list):
        return None
    return np.array(matrix_data, dtype=int)


def plot_confusion(matrix: np.ndarray, save_path: Path) -> None:
    col_sums = matrix.sum(axis=0, keepdims=True).astype(float)
    col_sums[col_sums == 0] = 1
    display_matrix = matrix.astype(float) / col_sums * 100.0

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(display_matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_yticklabels(CLASS_NAMES, fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel("Ground Truth", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted", fontsize=AXIS_LABEL_FONTSIZE, labelpad=PREDICTED_LABEL_PAD)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_color = "white" if i == j else "black"
            ax.text(
                j, i,
                f"{display_matrix[i, j]:.1f}",
                ha="center", va="center",
                color=text_color,
                fontsize=CELL_VALUE_FONTSIZE,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.unlink(missing_ok=True)  # delete old file first to force overwrite
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    # --- YOLO models ---
    all_model_dirs = {p.name: p for p in DETECT_DIR.iterdir() if p.is_dir() and (p / "eval_cache").exists()}
    ordered = [n for n in MODEL_ORDER if n in all_model_dirs]
    extras = [n for n in sorted(all_model_dirs) if n not in MODEL_ORDER]
    models = ordered + extras

    for model_name in models:
        cache_file = pick_cache_file(model_name)
        if cache_file is None:
            print(f"[SKIP] {model_name}: no eval_cache found")
            continue

        matrix = load_matrix(cache_file)
        if matrix is None:
            print(f"[SKIP] {model_name}: failed to load matrix from {cache_file.name}")
            continue

        save_path = WEIGHTS_DIR / f"{model_name}.png"
        plot_confusion(matrix, save_path)
        print(f"[OK]   {model_name}: saved → {save_path}")

    # --- Faster R-CNN ---
    if FASTERRCNN_CACHE_DIR.exists():
        # Prefer a file prefixed with "test_", fallback to most recent
        test_files = sorted(FASTERRCNN_CACHE_DIR.glob("test_*"))
        cache_file = test_files[-1] if test_files else max(
            FASTERRCNN_CACHE_DIR.iterdir(), key=lambda p: p.stat().st_mtime, default=None
        )
        if cache_file is not None:
            matrix = load_matrix(cache_file)
            if matrix is not None:
                save_path = WEIGHTS_DIR / "fasterrcnn.png"
                plot_confusion(matrix, save_path)
                print(f"[OK]   fasterrcnn: saved → {save_path}")
            else:
                print(f"[SKIP] fasterrcnn: failed to load matrix from {cache_file.name}")
        else:
            print(f"[SKIP] fasterrcnn: eval_cache is empty")
    else:
        print(f"[SKIP] fasterrcnn: cache dir not found ({FASTERRCNN_CACHE_DIR})")


if __name__ == "__main__":
    main()
