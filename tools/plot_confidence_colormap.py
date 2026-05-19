"""
Generate a wide PNG showing the confidence threshold colormap.
Useful as a reference for the PR curve color scheme.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "weights"

# ── Configuration ────────────────────────────────────────────────────────────
BAR_WIDTH_PIXELS = 1400      # Width of the color bar in pixels
BAR_HEIGHT_PIXELS = 1       # Height of the color bar in pixels
COLORMAP_NAME = "plasma"     # Matplotlib colormap (plasma, viridis, etc.)
DPI = 150                    # Output resolution
NUM_TICKS = 11               # Number of confidence value labels below the bar

def plot_confidence_colormap(
    conf_min: float = 0.001,
    conf_max: float = 0.999,
    width: int | None = None,
    height: int | None = None,
    colormap: str | None = None,
    output_path: Path | None = None,
):
    """
    Generate a wide PNG showing confidence threshold coloring (no text).
    
    Args:
        conf_min: Minimum confidence value
        conf_max: Maximum confidence value
        width: Width in pixels (uses BAR_WIDTH_PIXELS if None)
        height: Height in pixels (uses BAR_HEIGHT_PIXELS if None)
        colormap: Matplotlib colormap name (uses COLORMAP_NAME if None)
        output_path: Where to save the PNG (default: runs/detect/weights/)
    """
    if width is None:
        width = BAR_WIDTH_PIXELS
    if height is None:
        height = BAR_HEIGHT_PIXELS
    if colormap is None:
        colormap = COLORMAP_NAME
    
    if output_path is None:
        output_path = OUTPUT_DIR / f"confidence_colormap_{colormap}.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a 2D gradient: confidence values along x-axis
    gradient = np.linspace(conf_min, conf_max, width)
    gradient_2d = np.tile(gradient, (height, 1))
    
    # Plot (minimal: just the bar with ticks below)
    # Scale figure height to include space for tick labels (adaptive to bar height)
    label_space = max(40, int(height * 0.5))  # At least 40px or 50% of bar height
    fig_width = width / DPI
    fig_height = (height + label_space) / DPI
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI)
    
    ax.imshow(
        gradient_2d,
        aspect="auto",
        cmap=colormap,
        extent=[conf_min, conf_max, 0, 1],
        origin="lower",
    )
    
    # Add confidence tick labels at the bottom
    tick_positions = np.linspace(conf_min, conf_max, NUM_TICKS)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{v:.2f}" for v in tick_positions], fontsize=8)
    ax.set_yticks([])
    
    # Keep only the bottom spine for tick labels
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    # Adaptive bottom spacing based on bar height
    bottom_margin = max(0.15, label_space / (height + label_space) * 0.5)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=bottom_margin)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"✓ Saved confidence colormap: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate a minimal PNG showing confidence threshold coloring (bar with tick labels)"
    )
    parser.add_argument("--conf-min", type=float, default=0.001, help="Min confidence")
    parser.add_argument("--conf-max", type=float, default=0.999, help="Max confidence")
    parser.add_argument("--width", type=int, default=None, help="Image width (pixels, default: BAR_WIDTH_PIXELS)")
    parser.add_argument("--height", type=int, default=None, help="Image height (pixels, default: BAR_HEIGHT_PIXELS)")
    parser.add_argument("--colormap", type=str, default=None, help="Matplotlib colormap (default: COLORMAP_NAME)")
    parser.add_argument("--ticks", type=int, default=None, help="Number of confidence ticks (default: NUM_TICKS)")
    parser.add_argument("--dpi", type=int, default=None, help="Output DPI (default: DPI)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    # Override globals if provided
    if args.dpi is not None:
        DPI = args.dpi
    if args.ticks is not None:
        NUM_TICKS = args.ticks
    
    output_path = Path(args.output) if args.output else None
    
    plot_confidence_colormap(
        conf_min=args.conf_min,
        conf_max=args.conf_max,
        width=args.width,
        height=args.height,
        colormap=args.colormap,
        output_path=output_path,
    )
