"""Few labelled faults: plain kNN against the negative bank on the same splits.

Both scorers run on the same split of every category, seed and k (labelled faults per defect
type), so each negative-bank run has a kNN run with the identical test set, and the gain is the
paired difference. The negative bank here takes its patches from the ground-truth masks and
selects lambda per run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmad.report.runs import require_complete, seed_summary
from cmad.report.tables import mean_std, write_table

# metric column -> (table heading, decimals)
METRICS = {
    "image_auroc": ("image AUROC", 2),
    "aupro_0.3": ("AUPRO@0.3", 2),
    "fpr_at_tau_val": ("FPR at τ_val", 1),
    "recall_at_tau_val": ("recall at τ_val", 1),
}
METHODS = {"knn": "kNN", "neg": "neg. bank", "gain": "gain"}
_KEYS = ["bb", "k", "category", "seed"]
_BANK = ["selected_lambda", "bank_precision", "bank_recall"]


def negbank_runs(
    study: pd.DataFrame,
    source: str = "gt_mask",
    scale: str = "none",
    top_frac: float | None = None,
) -> pd.DataFrame:
    """The few-abnormal negative-bank runs of one source and one lambda unit, checked for
    completeness per backbone and k. `scale` is "none" for lambda in raw distance units and
    "val_median" for lambda in normalised units."""
    neg = study[
        (study["family"] == "few_abn")
        & (study["scorer"] == "negbank")
        & (study["source"] == source)
        & (study["scale"] == scale)
        & (
            study["top_frac"].isna()
            if top_frac is None
            else study["top_frac"] == top_frac
        )
    ]
    what = f"few-abnormal negative bank ({source}, {top_frac}, {scale})"
    require_complete(neg, ["bb", "k"], what=what)
    return neg


def paired_runs(
    study: pd.DataFrame,
    source: str = "gt_mask",
    scale: str = "none",
    top_frac: float | None = None,
) -> pd.DataFrame:
    """One row per category, seed and k that has both a kNN and a negative-bank run, with the
    metrics of each (`knn_*`, `neg_*`) and their difference (`gain_*`).

    A category without any defect type that can spare k support images has no negative bank at
    that k (its run falls back to kNN in every seed); it is left out of that k's rows.

    `top_frac` is the share of patches taken per support image, for the sources that have one."""
    few = study[study["family"] == "few_abn"]
    knn = few[few["scorer"] == "knn"]
    require_complete(knn, ["bb", "k"], what="few-abnormal kNN")
    neg = negbank_runs(study, source, scale, top_frac)

    columns = [*_KEYS, "split_sha1", *METRICS]
    pairs = knn[columns].merge(
        neg[[*columns, "neg_active", *_BANK]],
        on=_KEYS,
        suffixes=("_knn", "_neg"),
        validate="one_to_one",
    )
    if not (pairs["split_sha1_knn"] == pairs["split_sha1_neg"]).all():
        raise ValueError(
            "a kNN run and its negative-bank run were made on different splits"
        )

    active = pairs.groupby(["bb", "k", "category"], observed=True)["neg_active"].agg(
        ["sum", "size"]
    )
    mixed = active[(active["sum"] > 0) & (active["sum"] < active["size"])]
    if len(mixed):
        raise ValueError(f"negative bank active in some seeds only:\n{mixed}")
    with_bank = active[active["sum"] > 0].reset_index()[["bb", "k", "category"]]
    pairs = pairs.merge(with_bank, on=["bb", "k", "category"])

    out = pairs[[*_KEYS, *_BANK]].copy()
    for metric in METRICS:
        out[f"knn_{metric}"] = pairs[f"{metric}_knn"]
        out[f"neg_{metric}"] = pairs[f"{metric}_neg"]
        out[f"gain_{metric}"] = pairs[f"{metric}_neg"] - pairs[f"{metric}_knn"]
    return out


def few_abnormal_table(study: pd.DataFrame) -> pd.DataFrame:
    """Per backbone and k: mean ± std over seeds of the category mean, for kNN, the negative bank
    and the paired gain. Long format, one row per backbone, k and method."""
    pairs = paired_runs(study)
    parts = []
    for method in METHODS:
        columns = {f"{method}_{metric}": metric for metric in METRICS}
        summary = seed_summary(
            pairs.rename(columns=columns), ["bb", "k"], list(METRICS)
        )
        summary.insert(2, "method", method)
        parts.append(summary)
    table = pd.concat(parts, ignore_index=True)
    table["method"] = pd.Categorical(
        table["method"], categories=list(METHODS), ordered=True
    )
    return table.sort_values(["bb", "k", "method"]).reset_index(drop=True)


def write_few_abnormal(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    table = few_abnormal_table(study)
    header = [
        "backbone",
        "k",
        "categories",
        "method",
        *(title for title, _ in METRICS.values()),
    ]
    rows = []
    for record in table.to_dict("records"):
        first = record["method"] == "knn"
        cells = [
            str(record["bb"]) if first else "",
            str(record["k"]) if first else "",
            str(record["n_categories"]) if first else "",
            METHODS[record["method"]],
        ]
        for metric, (_, digits) in METRICS.items():
            cells.append(
                mean_std(
                    record[f"{metric}_mean"],
                    record[f"{metric}_std"],
                    digits,
                    signed=record["method"] == "gain",
                )
            )
        rows.append(cells)
    note = (
        "Percent. Mean ± standard deviation over 5 seeds of the mean over categories; the gain "
        "is the paired difference (same split, same test images). Negative patches from the "
        "ground-truth masks, λ selected per run. k is the number of labelled faults per defect "
        "type; the test set differs between k. At k = 10 three categories have no defect type "
        "with enough images to spare and are left out of both rows."
    )
    return write_table(
        out_dir, "few_abnormal", table, header, rows, "llrl" + "r" * len(METRICS), note
    )
