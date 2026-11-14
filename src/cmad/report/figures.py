"""Shared look of the figures: one colour per backbone, quiet axes, PDF and PNG output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no window is ever opened
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Colour names the backbone in every figure; what else varies in a panel (k, the method) is told
# apart by line style and a label, never by a second colour scale.
BACKBONE_COLOR = {"S": "#2a78d6", "B": "#eb6834", "WRN": "#1baf7a"}
BACKBONE_TITLE = {"S": "DINOv2-S", "B": "DINOv2-B", "WRN": "WRN-50"}
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

STYLE = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.labelcolor": SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY,
    "ytick.labelcolor": SECONDARY,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,  # keep text as text in the PDF
}


plt.rcParams.update(STYLE)


def panels(
    n: int, width: float = 6.3, height: float = 2.2, sharey: bool = False
) -> tuple[Figure, list]:
    """A row of `n` panels. 6.3 in is the text width of an A4 report."""
    figure, axes = plt.subplots(
        1, n, figsize=(width, height), squeeze=False, sharey=sharey
    )
    return figure, list(axes[0])


def save(figure: Figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    # leave a strip at the top for a figure-level legend, if there is one
    figure.tight_layout(rect=(0, 0, 1, 0.93 if figure.legends else 1))
    figure.savefig(paths[0])
    figure.savefig(paths[1], dpi=200)
    plt.close(figure)
    return paths
