"""Where the negative patches come from: ground-truth masks, the patches farthest from normal,
or patches drawn at random.

The mask source needs pixel labels. The other two need only the image-level label "faulty":
`top_q` takes the share q of a support image's patches that lie farthest from the normal bank,
`random` takes the same number at random and is the control for that choice. Every source is
compared at its own selected lambda, on the same splits, against plain kNN and against the masks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmad.report.few_abnormal import paired_runs
from cmad.report.runs import seed_summary
from cmad.report.tables import mean_std, write_table

FRACTIONS = (0.01, 0.05, 0.10, 0.25)
SOURCES = [("gt_mask", None)] + [(s, q) for s in ("top_q", "random") for q in FRACTIONS]
_VALUES = [
    "image_auroc",
    "gain",
    "vs_masks",
    "aupro",
    "aupro_gain",
    "bank_precision",
    "bank_recall",
]


def source_label(source: str, top_frac: float | None) -> str:
    if source == "gt_mask":
        return "masks"
    return f"{'farthest' if source == 'top_q' else 'random'} {100 * top_frac:g} %"


def neg_source_table(study: pd.DataFrame) -> pd.DataFrame:
    """Per backbone, k and source: image AUROC at the selected lambda, its paired gain over kNN
    and its paired difference to the mask source, the AUPRO gain, and how much of the bank lies
    on a defect. Mean ± std over seeds of the category mean."""
    keys = ["bb", "k", "category", "seed"]
    masks = paired_runs(study)[[*keys, "neg_image_auroc"]].rename(
        columns={"neg_image_auroc": "masks_image_auroc"}
    )
    parts = []
    for order, (source, top_frac) in enumerate(SOURCES):
        pairs = paired_runs(study, source=source, top_frac=top_frac).merge(
            masks, on=keys, validate="one_to_one"
        )
        runs = pd.DataFrame(
            {
                "bb": pairs["bb"],
                "k": pairs["k"],
                "category": pairs["category"],
                "seed": pairs["seed"],
                "image_auroc": pairs["neg_image_auroc"],
                "gain": pairs["gain_image_auroc"],
                "vs_masks": pairs["neg_image_auroc"] - pairs["masks_image_auroc"],
                "aupro": pairs["neg_aupro_0.3"],
                "aupro_gain": pairs["gain_aupro_0.3"],
                "bank_precision": 100.0 * pairs["bank_precision"],
                "bank_recall": 100.0 * pairs["bank_recall"],
            }
        )
        summary = seed_summary(runs, ["bb", "k"], _VALUES)
        summary.insert(2, "source", source_label(source, top_frac))
        summary.insert(3, "order", order)
        parts.append(summary)
    table = pd.concat(parts, ignore_index=True)
    return (
        table.sort_values(["bb", "k", "order"])
        .drop(columns="order")
        .reset_index(drop=True)
    )


def write_neg_source(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    table = neg_source_table(study)
    header = [
        "backbone",
        "k",
        "source",
        "image AUROC",
        "gain over kNN",
        "difference to masks",
        "AUPRO@0.3 gain",
        "bank on defect",
    ]
    rows = []
    for record in table.to_dict("records"):
        first = record["source"] == "masks"
        rows.append(
            [
                str(record["bb"]) if first else "",
                str(record["k"]) if first else "",
                record["source"],
                mean_std(record["image_auroc_mean"], record["image_auroc_std"], 2),
                mean_std(record["gain_mean"], record["gain_std"], 2, signed=True),
                ""
                if first
                else mean_std(record["vs_masks_mean"], record["vs_masks_std"], 2, True),
                mean_std(
                    record["aupro_gain_mean"], record["aupro_gain_std"], 2, signed=True
                ),
                f"{record['bank_precision_mean']:.0f} %",
            ]
        )
    note = (
        "Percent. Mean ± standard deviation over 5 seeds of the mean over categories, each source "
        "at its own selected λ. Gains are paired differences on the same split: over plain kNN, "
        "and to the mask source. 'Bank on defect' is the share of the negative bank's patches "
        "that lie on a ground-truth defect. At k = 10 the three categories without a negative "
        "bank are left out."
    )
    return write_table(out_dir, "neg_source", table, header, rows, "lllrrrrr", note)
