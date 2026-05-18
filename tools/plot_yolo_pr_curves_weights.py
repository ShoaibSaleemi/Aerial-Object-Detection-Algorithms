"""
PR-curve plotter that reads pre-computed sweep caches from
runs/detect/weights/ — no dataset or model inference required.

Auto-discovers sweep_cache_<model>_<split>_steps<n>_... .npz files
and generates a PR-curve PNG + PDF for each one found.

Usage
-----
    python tools/plot_yolo_pr_curves_weights.py                     # test split, 2000 steps
    python tools/plot_yolo_pr_curves_weights.py --split test2       # test2 split
    python tools/plot_yolo_pr_curves_weights.py --model yolo8n      # one model only
    python tools/plot_yolo_pr_curves_weights.py --split any         # all splits
    python tools/plot_yolo_pr_curves_weights.py --out-dir runs/detect/weights/plots
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR    = PROJECT_ROOT / "runs" / "detect" / "weights"

# ---------------------------------------------------------------------------
# Per-model confidence thresholds  (star marker on the PR curve)
# ---------------------------------------------------------------------------

MODEL_CONF_THRESH = {
    "yolo8n":  0.6863484706628682,
    "yolo8m":  0.7133918823950539,
    "yolo9t":  0.6724046133517759,
    "yolo10n": 0.5910035105688879,
    "yolo11n": 0.712997868833143,
    "yolo12n": 0.6838702654977842,
    "yolo26n": 0.6052508580184951,
}

MODEL_ORDER = ["yolo8n", "yolo8m", "yolo9t", "yolo10n", "yolo11n", "yolo12n", "yolo26n"]

# ---------------------------------------------------------------------------
# Plot style — edit these to change appearance without touching the logic
# ---------------------------------------------------------------------------

PLOT_FIGSIZE          = (7, 6)
PLOT_LINEWIDTH        = 2.5
PLOT_COLORMAP         = "plasma"
PLOT_SMOOTH_SIGMA     = 9.0       # Gaussian smoothing sigma (0 = off)

PLOT_XLIM             = (0.0, 1.0)
PLOT_YLIM             = (0.0, 1.05)

PLOT_XLABEL_FONTSIZE  = 20
PLOT_YLABEL_FONTSIZE  = 20
PLOT_TICK_FONTSIZE    = 20
PLOT_LEGEND_FONTSIZE  = 20

PLOT_GRID_ALPHA       = 0.3
PLOT_GRID_LINESTYLE   = "--"
PLOT_GRID_LINEWIDTH   = 0.7

PLOT_MARKER           = "$*$"     # best-point marker
PLOT_MARKER_SIZE      = 20
PLOT_MARKER_COLOR     = "black"

PLOT_DPI              = 150

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    radius = max(1, int(3 * sigma))
    x      = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    valid = ~(np.isnan(recall) | np.isnan(precision))
    r, p  = recall[valid], precision[valid]
    if len(r) < 2:
        return float("nan")
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------

# Pattern: sweep_cache_<model>_<split>_steps<n>_cmin<v>_cmax<v>.npz
_CACHE_RE = re.compile(
    r"^sweep_cache_(?P<model>\w+)_(?P<split>[^_]+)"
    r"_steps(?P<steps>\d+)_cmin(?P<cmin>[^_]+)_cmax(?P<cmax>[^_]+)\.npz$"
)


def discover_caches(
    cache_dir: Path,
    split_filter: str | None,
    steps_filter: int | None,
    model_filter: str | None,
) -> list[dict]:
    """Return sorted list of dicts: model, split, steps, path."""
    found = []
    for f in sorted(cache_dir.glob("sweep_cache_*.npz")):
        m = _CACHE_RE.match(f.name)
        if not m:
            continue
        model = m.group("model")
        split = m.group("split")
        steps = int(m.group("steps"))

        if model_filter and model != model_filter:
            continue
        if split_filter and split_filter != "any" and split != split_filter:
            continue
        if steps_filter and steps != steps_filter:
            continue

        found.append({"model": model, "split": split, "steps": steps, "path": f})

    # Sort by MODEL_ORDER, then split, then steps descending (prefer more steps)
    order_map = {n: i for i, n in enumerate(MODEL_ORDER)}
    found.sort(key=lambda x: (order_map.get(x["model"], 99), x["split"], -x["steps"]))
    return found


def deduplicate(entries: list[dict]) -> list[dict]:
    """Keep only the highest-step cache per (model, split) pair."""
    seen: set = set()
    unique: list = []
    for e in entries:
        key = (e["model"], e["split"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_and_save(
    model_name: str,
    split: str,
    conf_values: np.ndarray,
    macro_P: np.ndarray,
    macro_R: np.ndarray,
    best_conf: float | None,
    smooth_sigma: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    plot_R = _gaussian_smooth(macro_R, smooth_sigma)
    plot_P = _gaussian_smooth(macro_P, smooth_sigma)

    points   = np.column_stack([plot_R, plot_P]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=float(conf_values.min()), vmax=float(conf_values.max()))
    lc = LineCollection(segments, cmap=PLOT_COLORMAP, norm=norm,
                        linewidth=PLOT_LINEWIDTH, alpha=1.0, zorder=3)
    lc.set_array((conf_values[:-1] + conf_values[1:]) / 2.0)
    ax.add_collection(lc)

    if best_conf is not None:
        idx = int(np.argmin(np.abs(conf_values - best_conf)))
        ax.plot(
            float(plot_R[idx]), float(plot_P[idx]),
            marker=PLOT_MARKER, markersize=PLOT_MARKER_SIZE, color=PLOT_MARKER_COLOR,
            zorder=6, linestyle="None",
            label="Best",
        )
        ax.legend(loc="lower left", fontsize=PLOT_LEGEND_FONTSIZE, framealpha=0.85)

    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.set_xlabel("Recall", fontsize=PLOT_XLABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=PLOT_YLABEL_FONTSIZE)
    ax.grid(True, alpha=PLOT_GRID_ALPHA, linestyle=PLOT_GRID_LINESTYLE,
            linewidth=PLOT_GRID_LINEWIDTH)
    ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_pr_curve_{split}"
    for ext in (".png", ".pdf"):
        out = out_dir / f"{stem}{ext}"
        fig.savefig(out, dpi=PLOT_DPI)
        print(f"  Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot YOLO PR curves from pre-computed sweep caches (no dataset needed)"
    )
    parser.add_argument("--model",   type=str,   default=None,
                        help="Filter to this model name (e.g. yolo8n)")
    parser.add_argument("--split",   type=str,   default="test",
                        help="Dataset split to use (default: test). Use 'any' for all splits.")
    parser.add_argument("--steps",   type=int,   default=2000,
                        help="Prefer caches with this step count (default: 2000). 0 = any.")
    parser.add_argument("--smooth",  type=float, default=PLOT_SMOOTH_SIGMA,
                        help="Gaussian smoothing sigma (default: %(default)s, 0 = off)")
    parser.add_argument("--out-dir", type=str,   default=None,
                        help="Output directory for PNG/PDF (default: same as cache dir)")
    args = parser.parse_args()

    steps_filter = args.steps if args.steps > 0 else None
    out_dir      = Path(args.out_dir) if args.out_dir else CACHE_DIR

    print("=" * 70)
    print("  YOLO PR Curve Plotter  (from sweep cache — no dataset needed)")
    print("=" * 70)
    print(f"  Cache dir : {CACHE_DIR}")
    print(f"  Split     : {args.split}")
    print(f"  Steps     : {steps_filter or 'any'}")
    print(f"  Out dir   : {out_dir}")

    entries = discover_caches(CACHE_DIR, args.split, steps_filter, args.model)
    if not entries:
        print("\n  No matching sweep caches found.")
        print("  Try: --split any   or   --steps 0   or   check --model name")
        return

    entries = deduplicate(entries)

    print(f"\n  Found {len(entries)} cache(s):\n")
    for e in entries:
        print(f"    {e['path'].name}")

    print()
    ap_summary: dict = {}

    for e in entries:
        model_name = e["model"]
        split      = e["split"]
        cache_path = e["path"]

        print(f"{'─' * 70}")
        print(f"  {model_name}  [{split}]  ←  {cache_path.name}")

        d           = np.load(cache_path)
        conf_values = d["conf_values"]
        macro_P     = d["macro_P"]
        macro_R     = d["macro_R"]

        ap        = compute_ap(macro_R, macro_P)
        best_conf = MODEL_CONF_THRESH.get(model_name)

        ap_summary[model_name] = ap
        bc_str = f"{best_conf:.4f}" if best_conf is not None else "N/A (no star marker)"
        print(f"  AP (macro-avg): {ap:.4f}   best_conf: {bc_str}")

        plot_and_save(model_name, split, conf_values, macro_P, macro_R,
                      best_conf, args.smooth, out_dir)

    print(f"\n{'=' * 70}")
    print(f"  AP Summary  (split={args.split})")
    print(f"{'=' * 70}")
    order_map = {n: i for i, n in enumerate(MODEL_ORDER)}
    for name, ap_val in sorted(ap_summary.items(), key=lambda x: order_map.get(x[0], 99)):
        print(f"  {name:<15}  AP = {ap_val:.4f}")


if __name__ == "__main__":
    main()
