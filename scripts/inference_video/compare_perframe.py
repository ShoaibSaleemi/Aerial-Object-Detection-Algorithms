"""
Compare per-frame metrics from two inference runs on the same video.

Usage (positional):
    python compare_perframe.py <npz_a> <npz_b>

Or run without arguments to pick interactively from existing .npz files in
    runs/detect/inference_video/

Output:
    runs/detect/inference_video/<stem_a>_vs_<stem_b>_compare.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import questionary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = PROJECT_ROOT / "runs" / "detect" / "inference_video"

# Colours for the two runs (matplotlib colour strings)
COLOR_A = "#1f77b4"   # blue
COLOR_B = "#ff7f0e"   # orange


def choose_npz(prompt: str) -> Path:
    npz_files = sorted(OUTPUT_DIR.glob("*_perframe.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_perframe.npz files found in {OUTPUT_DIR}")
    name = questionary.select(prompt, choices=[p.name for p in npz_files]).ask()
    if not name:
        raise ValueError("No file selected.")
    return OUTPUT_DIR / name


def load_npz(path: Path) -> dict:
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def short_label(path: Path) -> str:
    """Strip the trailing _perframe suffix for a readable legend label."""
    stem = path.stem  # e.g. "visible_yolo8n_perframe"
    if stem.endswith("_perframe"):
        stem = stem[: -len("_perframe")]
    return stem


def main():
    if len(sys.argv) >= 3:
        path_a = Path(sys.argv[1])
        path_b = Path(sys.argv[2])
        if not path_a.exists():
            raise FileNotFoundError(f"File not found: {path_a}")
        if not path_b.exists():
            raise FileNotFoundError(f"File not found: {path_b}")
    else:
        path_a = choose_npz("Choose first run  (will be blue):")
        path_b = choose_npz("Choose second run  (will be orange):")

    da = load_npz(path_a)
    db = load_npz(path_b)

    label_a = short_label(path_a)
    label_b = short_label(path_b)

    # ── Subplots ──────────────────────────────────────────────────────────────
    metrics = [
        ("IoU",                    "frame_ious",    (0.0, 1.0)),
        ("Center distance (px)",   "frame_dists",   None),
        ("Detection confidence",   "frame_confs",   (0.0, 1.0)),
        ("Class ID",               "frame_cls_ids", (-1.5, 2.5)),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 3 * len(metrics)), sharex=False)
    fig.suptitle(f"{label_a}  vs  {label_b}  —  per-frame metrics", fontsize=13, fontweight="bold")

    for ax, (ylabel, key, ylim) in zip(axes, metrics):
        if key in da:
            ax.plot(da["frame_numbers"], da[key], linewidth=1.0,
                    color=COLOR_A, label=label_a, alpha=0.85)
        if key in db:
            ax.plot(db["frame_numbers"], db[key], linewidth=1.0,
                    color=COLOR_B, label=label_b, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.35)

    # Class ID axis: integer ticks with names
    cls_ax = axes[-1]
    cls_ax.set_yticks([-1, 0, 1, 2])
    cls_ax.set_yticklabels(["none", "bird", "drone", "unknown"], fontsize=8)

    axes[-1].set_xlabel("Frame", fontsize=10)
    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    stem_a = path_a.stem.replace("_perframe", "")
    stem_b = path_b.stem.replace("_perframe", "")
    out_path = OUTPUT_DIR / f"{stem_a}_vs_{stem_b}_compare.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
