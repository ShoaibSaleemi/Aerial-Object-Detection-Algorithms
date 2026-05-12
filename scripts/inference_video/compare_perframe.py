"""
Compare per-frame metrics from up to 4 inference runs on the same video.

Usage (positional):
    python compare_perframe.py <npz_1> [npz_2] [npz_3] [npz_4]

Or run without arguments to pick interactively (NUM_RUNS files will be asked).

Output:
    runs/detect/inference_video/<stems>_compare.png  (raster)
    runs/detect/inference_video/<stems>_compare.pdf  (vector)
"""

# ── Number of runs to compare (1 – 4) ─────────────────────────────────────────
NUM_RUNS = 2

# ── Plot appearance ────────────────────────────────────────────────────────────
FIG_WIDTH        = 13    # figure width in inches
FIG_HEIGHT_PER   = 3     # height per subplot in inches
PNG_DPI          = 150   # raster resolution

FONT_TITLE       = 18    # suptitle font size
FONT_YLABEL      = 15    # y-axis label font size
FONT_XLABEL      = 15    # x-axis label font size
FONT_LEGEND      = 15    # legend font size
FONT_CLS_TICKS   = 10    # class-ID tick label font size

LINE_WIDTH       = 1.0   # plot line width
LINE_ALPHA       = 0.85  # line opacity
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import questionary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = PROJECT_ROOT / "runs" / "detect" / "inference_video"

# Colours for up to 4 runs (matplotlib colour strings)
COLOR_A = "#1f77b4"   # blue
COLOR_B = "#ff7f0e"   # orange
COLOR_C = "#2ca02c"   # green
COLOR_D = "#d62728"   # red

_COLORS  = [COLOR_A, COLOR_B, COLOR_C, COLOR_D]
_COLOR_NAMES = ["blue", "orange", "green", "red"]


def choose_npz(prompt: str, color: str) -> Path:
    npz_files = sorted(OUTPUT_DIR.glob("*_perframe.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_perframe.npz files found in {OUTPUT_DIR}")
    style = questionary.Style([("question", f"fg:{color} bold"), ("answer", f"fg:{color} bold")])
    name = questionary.select(prompt, choices=[p.name for p in npz_files], style=style).ask()
    if not name:
        raise ValueError("No file selected.")
    return OUTPUT_DIR / name


def _validate_num_runs(n: int) -> int:
    if not (1 <= n <= 4):
        raise ValueError(f"NUM_RUNS must be 1–4, got {n}")
    return n


def load_npz(path: Path) -> dict:
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


_DISPLAY_NAMES = {
    "wbf":     "WBF",
    "yolo8n":  "YOLOv8n",
    "yolo8m":  "YOLOv8m",
    "yolo8s":  "YOLOv8s",
    "yolo9t":  "YOLOv9t",
    "yolo10n": "YOLOv10n",
    "yolo11n": "YOLOv11n",
    "yolo12n": "YOLOv12n",
    "yolo26n": "YOLO26n",
}


def short_label(path: Path) -> str:
    """Return a display name derived from the run name embedded in the file stem.

    File stems follow the pattern  {video_stem}_{run_name}_perframe,
    so the run name is the second-to-last underscore-delimited token.
    """
    stem = path.stem  # e.g. "visible_yolo8n_perframe"
    if stem.endswith("_perframe"):
        stem = stem[: -len("_perframe")]
    # Last token after stripping _perframe is the run name
    run_name = stem.rsplit("_", 1)[-1]
    return _DISPLAY_NAMES.get(run_name, run_name)


def main():
    n = _validate_num_runs(NUM_RUNS)

    # ── Collect paths ──────────────────────────────────────────────────────────
    cli_paths = [Path(p) for p in sys.argv[1:] if p.endswith(".npz")]
    paths: list[Path] = []

    if len(cli_paths) >= n:
        paths = cli_paths[:n]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
    else:
        ordinals = ["first", "second", "third", "fourth"]
        for i in range(n):
            color_name = _COLOR_NAMES[i]
            p = choose_npz(
                f"Choose {ordinals[i]} run  (will be {color_name}):",
                _COLORS[i],
            )
            paths.append(p)

    datasets = [load_npz(p) for p in paths]
    labels   = [short_label(p) for p in paths]

    # ── Subplots ──────────────────────────────────────────────────────────────
    def _eiou_penalty(d: dict) -> np.ndarray | None:
        if "frame_ious" in d and "frame_eious" in d:
            return d["frame_ious"].astype(np.float32) - d["frame_eious"].astype(np.float32)
        return None

    metrics = [
        ("IoU",                    "frame_ious",    (0.0, 1.0)),
        ("EIoU penalty (IoU−EIoU)", None,           (0.0, None)),
        ("Detection confidence",   "frame_confs",   (0.0, 1.0)),
        ("Class ID",               "frame_cls_ids", (-1.5, 2.5)),
    ]

    fig, axes = plt.subplots(
        len(metrics), 1,
        figsize=(FIG_WIDTH, FIG_HEIGHT_PER * len(metrics)),
        sharex=False,
    )
    title = "  vs  ".join(labels) + "  —  per-frame metrics"
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold")

    for ax, (ylabel, key, ylim) in zip(axes, metrics):
        for d, label, color in zip(datasets, labels, _COLORS):
            vals = _eiou_penalty(d) if key is None else (d[key] if key in d else None)
            if vals is not None:
                ax.plot(d["frame_numbers"], vals, linewidth=LINE_WIDTH,
                        color=color, label=label, alpha=LINE_ALPHA)
        ax.set_ylabel(ylabel, fontsize=FONT_YLABEL)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=FONT_LEGEND, loc="upper right")
        ax.grid(True, alpha=0.35)

    cls_ax = axes[-1]
    cls_ax.set_yticks([-1, 0, 1, 2])
    cls_ax.set_yticklabels(["none", "bird", "drone", "unknown"], fontsize=FONT_CLS_TICKS)

    axes[-1].set_xlabel("Frame", fontsize=FONT_XLABEL)
    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    stems = "_vs_".join(p.stem.replace("_perframe", "") for p in paths)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUT_DIR / f"{stems}_compare.png"
    out_pdf = OUTPUT_DIR / f"{stems}_compare.pdf"
    fig.savefig(str(out_png), dpi=PNG_DPI)
    fig.savefig(str(out_pdf))
    plt.close(fig)
    print(f"Saved → {out_png}")
    print(f"Saved → {out_pdf}")


if __name__ == "__main__":
    main()
