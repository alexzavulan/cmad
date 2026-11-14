"""Results that use normal images only: the full training set (one-class) and k normal images."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmad.report.runs import require_complete, seed_summary
from cmad.report.tables import mean_std, write_table

# metric column -> (table heading, decimals)
ONE_CLASS_METRICS = {
    "image_auroc": ("image AUROC", 2),
    "image_ap": ("image AP", 2),
    "pixel_auroc": ("pixel AUROC", 2),
    "aupro_0.3": ("AUPRO@0.3", 2),
    "fpr_at_tau_val": ("FPR at τ_val", 1),
    "recall_at_tau_val": ("recall at τ_val", 1),
    "f1_at_tau_val": ("F1 at τ_val", 1),
}
FEW_NORMAL_METRICS = {"image_auroc": "image AUROC", "aupro_0.3": "AUPRO@0.3"}
_NOTE = "Percent. Mean ± standard deviation over 5 seeds of the mean over the 15 categories."


def _one_class(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs[(runs["family"] == "oneclass") & (runs["scorer"] == "knn")]
    require_complete(runs, ["bb", "layers"], what="one-class")
    return runs


def one_class_table(
    study: pd.DataFrame, other_layers: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per backbone and layer. `study` holds the main layers; `other_layers` may add the study
    runs of further layers (the backbones it does not cover simply have no such row)."""
    runs = _one_class(study)
    if other_layers is not None and len(other_layers):
        runs = pd.concat([runs, _one_class(other_layers)])
    table = seed_summary(runs, ["bb", "layers"], list(ONE_CLASS_METRICS))
    return table.sort_values(["bb", "layers"], ascending=[True, False]).reset_index(
        drop=True
    )


def per_category_table(study: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One-class, main layers: mean and std over seeds per category and backbone."""
    grouped = _one_class(study).groupby(["category", "bb"], observed=True)[metric]
    return grouped.agg(["mean", "std"]).reset_index()


def few_normal_table(study: pd.DataFrame) -> pd.DataFrame:
    """Per backbone and number of normal images k; the full training set is the last column
    (k empty), taken from the one-class runs."""
    few = study[(study["family"] == "few_norm") & (study["scorer"] == "knn")]
    require_complete(few, ["bb", "k"], what="few-normal")
    runs = pd.concat([few, _one_class(study)])
    return seed_summary(runs, ["bb", "k"], list(FEW_NORMAL_METRICS))


def write_normal_only(
    study: pd.DataFrame, other_layers: pd.DataFrame | None, out_dir: Path
) -> list[Path]:
    paths = []

    table = one_class_table(study, other_layers)
    header = ["backbone", "layer", *(title for title, _ in ONE_CLASS_METRICS.values())]
    rows = [
        [
            str(r["bb"]),
            r["layers"],
            *(
                mean_std(r[f"{metric}_mean"], r[f"{metric}_std"], digits)
                for metric, (_, digits) in ONE_CLASS_METRICS.items()
            ),
        ]
        for r in table.to_dict("records")
    ]
    align = "ll" + "r" * len(ONE_CLASS_METRICS)
    paths += write_table(out_dir, "one_class", table, header, rows, align, _NOTE)

    backbones = [str(b) for b in table["bb"].unique()]
    for metric, title in FEW_NORMAL_METRICS.items():
        name = metric.split("_0")[0]
        per_category = per_category_table(study, metric)
        wide = per_category.pivot(
            index="category", columns="bb", values=["mean", "std"]
        )
        rows = [
            [
                category,
                *(
                    mean_std(
                        wide.loc[category, ("mean", b)],
                        wide.loc[category, ("std", b)],
                        1,
                    )
                    for b in backbones
                ),
            ]
            for category in wide.index
        ]
        note = f"One-class {title}, percent. Mean ± standard deviation over 5 seeds."
        paths += write_table(
            out_dir,
            f"one_class_per_category_{name}",
            per_category,
            ["category", *backbones],
            rows,
            "l" + "r" * len(backbones),
            note,
        )

        few = few_normal_table(study)
        ks = sorted(k for k in few["k"].dropna().unique())
        cell = {(str(r["bb"]), r["k"]): r for r in few.to_dict("records")}
        rows = []
        for b in backbones:
            row = [b]
            for k in [*ks, pd.NA]:
                r = next(
                    v
                    for (bb, kk), v in cell.items()
                    if bb == b and (pd.isna(kk) if pd.isna(k) else kk == k)
                )
                row.append(mean_std(r[f"{metric}_mean"], r[f"{metric}_std"], 2))
            rows.append(row)
        header = ["backbone", *(f"k = {k}" for k in ks), "all"]
        note = (
            f"{title}, percent, with k normal training images ('all': the full training set). "
            + _NOTE.split(". ", 1)[1]
        )
        paths += write_table(
            out_dir,
            f"few_normal_{name}",
            few,
            header,
            rows,
            "l" + "r" * (len(ks) + 1),
            note,
        )
    return paths
