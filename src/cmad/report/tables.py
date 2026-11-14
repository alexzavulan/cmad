"""Writing a result table three ways: CSV (full precision), Markdown (to read) and LaTeX."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# Applied in order: the symbols with a subscript first, then the characters LaTeX reserves
_LATEX_ESCAPES = {
    "τ_val": r"$\tau_{\mathrm{val}}$",
    "λ": r"$\lambda$",
    "%": r"\%",
    "_": r"\_",
    "&": r"\&",
    "#": r"\#",
    "±": r"$\pm$",
    "−": "$-$",
}


def mean_std(mean: float, std: float, digits: int, signed: bool = False) -> str:
    """'99.18 ± 0.03'; with `signed`, the mean carries its sign (a typographic minus)."""
    if pd.isna(mean):
        return "—"
    mean = round(mean, digits) + 0.0  # no "-0.00"
    text = f"{mean:+.{digits}f}" if signed else f"{mean:.{digits}f}"
    return f"{text} ± {std:.{digits}f}".replace("-", "−")


def markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _latex_cell(text: str) -> str:
    # one pass over the text, so that an inserted "\tau_{...}" is not escaped again
    pattern = "|".join(re.escape(key) for key in _LATEX_ESCAPES)
    return re.sub(pattern, lambda match: _LATEX_ESCAPES[match[0]], text)


def latex(header: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> str:
    """A booktabs `tabular` (no float, no caption) to be pulled into the report with \\input."""
    body = [" & ".join(_latex_cell(c) for c in row) + r" \\" for row in rows]
    return (
        "\n".join(
            [
                rf"\begin{{tabular}}{{{align}}}",
                r"\toprule",
                " & ".join(_latex_cell(h) for h in header) + r" \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n"
    )


def write_table(
    out_dir: Path,
    name: str,
    data: pd.DataFrame,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: str,
    note: str = "",
) -> list[Path]:
    """`data` goes to <name>.csv untouched; `header` and `rows` are the formatted view, written
    to <name>.md (with `note` underneath) and <name>.tex."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.{ext}" for ext in ("csv", "md", "tex")]
    data.to_csv(paths[0], index=False)
    paths[1].write_text(markdown(header, rows) + (f"\n{note}\n" if note else ""))
    paths[2].write_text(latex(header, rows, align))
    return paths
