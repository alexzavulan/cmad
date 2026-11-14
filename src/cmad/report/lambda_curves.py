"""The negative-bank weight lambda: the curve over the whole grid, and ways of choosing it.

Every negative-bank run stores its per-image scores at each lambda of the grid, for the test
images and for the validation normals, so any rule for choosing lambda can be scored afterwards
without a new run. The threshold is re-set at each lambda (the largest validation-normal score
at that lambda). Rules compared, all at image level:

  knn        lambda = 0
  selected   the run's own choice, from its leave-one-support-out folds (the method)
  shared     one lambda for all categories: for each category, the lambda that is best on the
             fold curves of the other categories of the same seed and k
  shared_k   the same, with the fold curves pooled over k = 1 and k = 5
  fixed_1    lambda = 1 with no selection (normalised units only)

Lambda is in raw distance units or, for the runs made with the validation-median scale, in
normalised units; the grids differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cmad.report.few_abnormal import negbank_runs
from cmad.report.figures import (
    AXIS,
    BACKBONE_COLOR,
    BACKBONE_TITLE,
    SECONDARY,
    panels,
    save,
)
from cmad.report.neg_source import source_label
from cmad.report.runs import order_backbones, seed_summary
from cmad.report.tables import mean_std, write_table

# (source, share of patches, scale) of the run sets that have curves to show
SETTINGS = [
    ("gt_mask", None, "none"),
    ("gt_mask", None, "val_median"),
    ("top_q", 0.10, "val_median"),
]
UNIT = {"none": "raw", "val_median": "normalised"}
POOLED_K = (1, 5)
RULES = {
    "knn": "kNN (λ = 0)",
    "selected": "selected per run",
    "shared": "shared across categories",
    "shared_k": "shared across categories and k",
    "fixed_1": "fixed λ = 1",
}
HARM_PT = 0.1  # a run counts as harmed when it is more than this below kNN
_VALUES = ["image_auroc", "fpr", "recall"]
_LINES = {1: "-", 5: "--", 10: ":"}


def _key(lam: float) -> str:
    return f"{lam:g}"


def lambda_runs(
    study: pd.DataFrame, source: str, scale: str, top_frac: float | None = None
) -> pd.DataFrame:
    """One row per run and lambda: image AUROC, and realised false-positive rate and recall at
    that lambda's threshold, in percent. `fold` is the run's leave-one-support-out curve (NaN
    where the run had too few support images to make folds) and `selected` marks its choice.
    Runs without a negative bank are left out."""
    runs = negbank_runs(study, source, scale, top_frac)
    rows = []
    for run in runs[runs["neg_active"].astype(bool)].itertuples():
        record = json.loads(Path(run.path).read_text())
        bank, scores = record["neg_bank"], record["image_scores"]
        labels = np.array(scores["test"]["label"])
        fold = bank["selection_rule"].get("curve") or {}
        chosen = _key(bank["selected_lambda"] or 0.0)
        for lam in sorted(bank["lambdas"]):
            key = _key(lam)
            test = np.array(scores["test"]["curve"][key])
            flagged = test > np.max(scores["val"]["curve"][key])
            rows.append(
                {
                    "bb": str(run.bb),
                    "k": int(run.k),
                    "category": run.category,
                    "seed": int(run.seed),
                    "lam": float(lam),
                    "image_auroc": 100.0 * bank["curve"][key]["image_auroc"],
                    "fpr": 100.0 * float(flagged[labels == 0].mean()),
                    "recall": 100.0 * float(flagged[labels == 1].mean()),
                    "fold": fold.get(key, np.nan),
                    "selected": key == chosen,
                }
            )
    return order_backbones(pd.DataFrame(rows))


def _best_elsewhere(folds: pd.DataFrame, category: str) -> float | None:
    """The lambda with the highest mean fold AUROC over the other categories' curves (each
    curve weighs the same); the smallest such lambda on a tie."""
    others = folds[(folds["category"] != category) & folds["fold"].notna()]
    if others.empty:
        return None
    mean = others.groupby("lam")["fold"].mean().sort_index()
    return float(
        mean.idxmax()
    )  # idxmax returns the first maximum, i.e. the smallest lambda


def rule_runs(curves: pd.DataFrame, fixed: float | None = None) -> pd.DataFrame:
    """One row per run and rule, with the metrics at the lambda that the rule gives the run and
    the image AUROC difference to kNN on the same run (`vs_knn`). `fixed` adds the rule that
    gives every run that lambda; it only means something in normalised units."""
    at = curves.set_index(["bb", "k", "category", "seed", "lam"])[_VALUES]
    rows = []
    for (bb, seed), group in curves.groupby(["bb", "seed"], observed=True):
        pooled = group[group["k"].isin(POOLED_K)]
        for (k, category), run in group.groupby(["k", "category"], observed=True):
            choice = {
                "knn": 0.0,
                "selected": float(run.loc[run["selected"], "lam"].iloc[0]),
                "shared": _best_elsewhere(group[group["k"] == k], category),
                "shared_k": _best_elsewhere(pooled, category)
                if k in POOLED_K
                else None,
                "fixed_1": fixed,
            }
            for rule, lam in choice.items():
                if lam is None:
                    continue
                values = at.loc[(bb, k, category, seed, lam)]
                rows.append(
                    {
                        "bb": bb,
                        "k": k,
                        "category": category,
                        "seed": seed,
                        "rule": rule,
                        "lam": lam,
                        **values,
                    }
                )
    out = pd.DataFrame(rows)
    knn = out[out["rule"] == "knn"].set_index(["bb", "k", "category", "seed"])[
        "image_auroc"
    ]
    out["vs_knn"] = (
        out["image_auroc"]
        - knn.reindex(
            pd.MultiIndex.from_frame(out[["bb", "k", "category", "seed"]])
        ).to_numpy()
    )
    return order_backbones(out)


def rule_table(by_rule: pd.DataFrame) -> pd.DataFrame:
    """Per backbone, k and rule: mean ± std over seeds of the category mean, the number of runs
    harmed (more than 0.1 pt below kNN) and the worst run."""
    table = seed_summary(by_rule, ["bb", "k", "rule"], _VALUES)
    harm = by_rule.groupby(["bb", "k", "rule"], observed=True)["vs_knn"].agg(
        n_harmed=lambda d: int((d < -HARM_PT).sum()), worst=("min")
    )
    table = table.merge(harm.reset_index(), on=["bb", "k", "rule"])
    table["rule"] = pd.Categorical(table["rule"], categories=list(RULES), ordered=True)
    return table.sort_values(["bb", "k", "rule"]).reset_index(drop=True)


def curve_figure(curve: pd.DataFrame, unit: str, ks: tuple[int, ...] = (1, 5, 10)):
    """Mean image AUROC against lambda, one panel per backbone, one line per k in `ks`. The grid
    points are spaced evenly. Both axes are cut: far past its optimum a curve drops by tens of
    points, which would flatten everything of interest, so the y-axis ends a little below kNN
    and the x-axis at the first grid point where every line has left the panel."""
    curve = curve[curve["k"].isin(ks)]
    backbones = [str(b) for b in curve["bb"].unique()]
    figure, axes = panels(len(backbones))
    grid = sorted(curve["lam"].unique())
    position = {lam: i for i, lam in enumerate(grid)}
    for axis, bb in zip(axes, backbones, strict=True):
        rows = curve[curve["bb"] == bb]
        floor = rows[rows["lam"] == 0.0]["image_auroc_mean"].min() - 0.5
        for k, line in rows.groupby("k"):
            line = line.sort_values("lam")
            x = [position[lam] for lam in line["lam"]]
            axis.plot(
                x,
                line["image_auroc_mean"],
                _LINES.get(int(k), "-."),
                color=BACKBONE_COLOR[bb],
                marker="o",
                markersize=2.5,
            )
        if unit == "normalised":
            axis.axvline(position[1.0], color=AXIS, linewidth=0.8, zorder=0)
        visible = rows[rows["image_auroc_mean"] >= floor]["lam"].max()
        last = min(position[visible] + 1, len(grid) - 1)
        axis.set_xlim(-0.4, last + 0.2)
        axis.set_ylim(floor, rows["image_auroc_mean"].max() + 0.1)
        axis.set_xticks(
            range(last + 1), [_key(lam) for lam in grid[: last + 1]], fontsize=6.5
        )
        axis.set_title(BACKBONE_TITLE[bb], color=SECONDARY, loc="left")
        axis.set_xlabel(f"λ ({unit} units)")
    axes[0].set_ylabel("image AUROC (%)")
    # the line styles mean the same in every panel: one legend, in neutral ink
    present = sorted(int(k) for k in curve["k"].unique())
    handles = [
        Line2D([], [], linestyle=_LINES.get(k, "-."), color=SECONDARY, label=f"k = {k}")
        for k in present
    ]
    figure.legend(
        handles=handles,
        loc="upper right",
        ncol=len(handles),
        bbox_to_anchor=(0.995, 1.0),
    )
    return figure


def write_lambda(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths, tables = [], []
    for source, top_frac, scale in SETTINGS:
        negatives, unit = source_label(source, top_frac), UNIT[scale]
        curves = lambda_runs(study, source, scale, top_frac)
        name = f"lambda_curve_{negatives.replace(' %', '').replace(' ', '')}_{unit}"
        curve = seed_summary(curves, ["bb", "k", "lam"], _VALUES)
        curve.to_csv(out_dir / f"{name}.csv", index=False)
        paths += [
            out_dir / f"{name}.csv",
            *save(curve_figure(curve, unit), out_dir, name),
        ]
        table = rule_table(
            rule_runs(curves, fixed=1.0 if unit == "normalised" else None)
        )
        table.insert(0, "negatives", negatives)
        table.insert(1, "units", unit)
        tables.append(table)

    table = pd.concat(tables, ignore_index=True)
    header = [
        "negatives",
        "λ units",
        "backbone",
        "k",
        "rule",
        "image AUROC",
        "FPR at τ_val",
        "recall at τ_val",
        "runs below kNN",
        "worst run",
    ]
    rows = []
    for r in table.to_dict("records"):
        first = r["rule"] == "knn"
        rows.append(
            [
                r["negatives"] if first else "",
                r["units"] if first else "",
                str(r["bb"]) if first else "",
                str(r["k"]) if first else "",
                RULES[r["rule"]],
                mean_std(r["image_auroc_mean"], r["image_auroc_std"], 2),
                mean_std(r["fpr_mean"], r["fpr_std"], 1),
                mean_std(r["recall_mean"], r["recall_std"], 1),
                "" if first else f"{r['n_harmed']} of {r['n_runs']}",
                "" if first else f"{r['worst']:+.2f}".replace("-", "−"),
            ]
        )
    note = (
        "Percent. Mean ± standard deviation over 5 seeds of the mean over categories; the "
        "threshold τ_val is re-set at each λ. 'Runs below kNN' counts runs more than 0.1 pt of "
        "image AUROC below the same run at λ = 0. The shared rules pick, for each category, the "
        "λ that is best on the leave-one-support-out curves of the other categories (and, for "
        "'and k', of k = 1 and 5 together). At k = 10 the categories without a negative bank "
        "are left out."
    )
    return paths + write_table(
        out_dir, "lambda_rules", table, header, rows, "lllll" + "r" * 5, note
    )
