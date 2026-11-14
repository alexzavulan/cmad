"""Shared look of the figures, made to sit in the LaTeX report.

The report is set in Palatino (the `mathpazo` package), so the figures use Palatino for text and
for mathematics too, at sizes close to the report's caption text when a figure is included at
the full text width. Axes are closed boxes with inward ticks, as in most LaTeX plots. One colour
per backbone; anything else that varies within a panel (k, seen or unseen) is told apart by line
style and the legend, never by a second colour scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no window is ever opened
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

BACKBONE_COLOR = {"S": "#2a78d6", "B": "#eb6834", "WRN": "#1baf7a"}
BACKBONE_TITLE = {"S": "DINOv2-S", "B": "DINOv2-B", "WRN": "WRN-50"}
INK = "#000000"
SECONDARY = "#000000"  # labels and titles are black, as in the text
REFERENCE = "#8c8c8c"  # reference lines (zero, lambda = 1)

# Palatino first (macOS); TeX Gyre Pagella and Book Antiqua are the same design elsewhere
SERIF = [
    "Palatino",
    "TeX Gyre Pagella",
    "Palatino Linotype",
    "Book Antiqua",
    "DejaVu Serif",
]

STYLE = {
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "custom",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "mathtext.sf": "serif",
    "mathtext.tt": "serif",
    "mathtext.cal": "serif:italic",  # unused; set so no other font is looked up
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "axes.formatter.use_mathtext": True,  # minus signs and exponents in the text font
    "pdf.fonttype": 42,  # keep text as text in the PDF
    "savefig.bbox": "tight",  # never crop a label: the saved area grows to fit the text
    "savefig.pad_inches": 0.03,
}

plt.rcParams.update(STYLE)


def panels(
    n: int, width: float = 6.3, height: float = 2.4, sharey: bool = False
) -> tuple[Figure, list]:
    """A row of `n` panels, laid out so that labels, titles and a figure legend never overlap.
    6.3 in is the text width of an A4 report, so the figure is included at its own size."""
    figure, axes = plt.subplots(
        1,
        n,
        figsize=(width, height),
        squeeze=False,
        sharey=sharey,
        layout="constrained",
    )
    return figure, list(axes[0])


def panel_title(axis, letter: str, text: str) -> None:
    """'(a) DINOv2-S' at the top left of a panel, as sub-figures are labelled in the report."""
    axis.set_title(f"({letter}) {text}", loc="left")


def reference_line(axis, value: float, vertical: bool = False) -> None:
    line = axis.axvline if vertical else axis.axhline
    line(value, color=REFERENCE, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)


def save(figure: Figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    figure.savefig(paths[0])
    figure.savefig(paths[1], dpi=200)
    plt.close(figure)
    return paths
