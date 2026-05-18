"""
Read WC-NMS eval_cache JSON files from runs/detect/<model>/eval_cache/
and generate percentage-normalised confusion matrix PNGs for each model.
Output is saved to runs/eval_wcnms/<model>_WC-NMS.png (overwrites existing).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECT_DIR = PROJECT_ROOT / "runs" / "detect"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "eval_wcnms"

CLASS_NAMES = ["bird", "drone", "unknown"]

MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n"]

# Visual settings — identical to eval_yolo.py / plot_confusion_from_cache.py
TICK_LABEL_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
CELL_VALUE_FONTSIZE = 33
PREDICTED_LABEL_PAD = -14


def pick_wcnms_cache_file(model_name: str) -> Path | None:
    """Return the most recent test_wcnms_metrics_* cache file for the model."""
    cache_dir = DETECT_DIR / model_name / "eval_cache"
    if not cache_dir.exists():
        return None
    test_files = sorted(cache_dir.glob("test_wcnms_metrics_*"), key=lambda p: p.stat().st_mtime)
    if test_files:
        return test_files[-1]
    # Fallback: any wcnms file
    all_files = sorted(cache_dir.glob("wcnms_metrics_*"), key=lambda p: p.stat().st_mtime)
    return all_files[-1] if all_files else None


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
            text_color = "white" if display_matrix[i, j] > 50 else "black"
            ax.text(
                j, i,
                f"{display_matrix[i, j]:.1f}",
                ha="center", va="center",
                color=text_color,
                fontsize=CELL_VALUE_FONTSIZE,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.unlink(missing_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    all_model_dirs = {
        p.name: p for p in DETECT_DIR.iterdir()
        if p.is_dir() and (p / "eval_cache").exists()
    }
    ordered = [n for n in MODEL_ORDER if n in all_model_dirs]
    extras = [n for n in sorted(all_model_dirs) if n not in MODEL_ORDER]
    models = ordered + extras

    for model_name in models:
        cache_file = pick_wcnms_cache_file(model_name)
        if cache_file is None:
            print(f"[SKIP] {model_name}: no WC-NMS eval_cache found")
            continue

        matrix = load_matrix(cache_file)
        if matrix is None:
            print(f"[SKIP] {model_name}: failed to load matrix from {cache_file.name}")
            continue

        save_path = OUTPUT_DIR / f"{model_name}_WC-NMS.png"
        plot_confusion(matrix, save_path)
        print(f"[OK]   {model_name}: saved -> {save_path}")


if __name__ == "__main__":
    main()
