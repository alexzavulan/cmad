"""Gain of the negative bank against the number of labelled faults per defect type (k).

A larger k moves more faulty images from the test set into the support set, and defect types
with too few images get no support at all, so the natural test set is different at every k. The
curve is therefore scored on one fixed set of images: those still in the test set at the largest
k, without the faulty images of defect types that have no support there. Each k keeps its own
run (bank, selected lambda, threshold); only the images it is scored on are restricted. kNN is
the same run at lambda = 0, scored on the same images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cmad.report.few_abnormal import negbank_runs
from cmad.report.figures import (
    AXIS,
    BACKBONE_COLOR,
    BACKBONE_TITLE,
    SECONDARY,
    panels,
    save,
)
from cmad.report.runs import order_backbones, seed_summary
from cmad.report.tables import mean_std, write_table

_VALUES = [
    "knn_auroc",
    "neg_auroc",
    "gain_auroc",
    "knn_recall",
    "neg_recall",
    "gain_recall",
]


def fixed_population(record: dict) -> set[str] | None:
    """The test images of a run at the largest k that the whole curve is scored on: every normal
    image, and the faulty images of the defect types that have support. None if the run has no
    negative bank."""
    bank = record["neg_bank"]
    if not bank["active"]:
        return None
    supported = {image["defect_type"] for image in bank["support"].values()}
    test = record["image_scores"]["test"]
    return {
        image
        for image, label, defect in zip(
            test["image_id"], test["label"], test["defect_type"], strict=True
        )
        if label == 0 or defect in supported
    }


def _score(record: dict, images: set[str]) -> dict[str, float]:
    scores = record["image_scores"]
    test, val = scores["test"], scores["val"]
    keep = np.array([image in images for image in test["image_id"]])
    if keep.sum() != len(images):
        raise ValueError(
            f"{record['category']} {record['regime']}: fixed-population images missing"
        )
    labels = np.array(test["label"])[keep]
    out = {"n_faulty": int(labels.sum()), "n_normal": int((labels == 0).sum())}
    for name, test_scores, val_scores in (
        ("neg", test["score"], val["score"]),
        ("knn", test["curve"]["0"], val["curve"]["0"]),
    ):
        kept = np.array(test_scores)[keep]
        out[f"{name}_auroc"] = 100.0 * roc_auc_score(labels, kept)
        out[f"{name}_recall"] = 100.0 * float(
            (kept[labels == 1] > np.max(val_scores)).mean()
        )
    out["gain_auroc"] = out["neg_auroc"] - out["knn_auroc"]
    out["gain_recall"] = out["neg_recall"] - out["knn_recall"]
    return out


def shots_runs(study: pd.DataFrame) -> pd.DataFrame:
    """One row per backbone, category, seed and k, scored on the fixed population of that
    backbone, category and seed. Categories without a negative bank at the largest k are left
    out. Negative patches from the ground-truth masks, lambda in raw units."""
    runs = negbank_runs(study)
    largest = int(runs["k"].max())
    rows = []
    for (bb, category, seed), group in runs.groupby(
        ["bb", "category", "seed"], observed=True
    ):
        records = {
            int(k): json.loads(Path(path).read_text())
            for k, path in zip(group["k"], group["path"], strict=True)
        }
        images = fixed_population(records[largest])
        if images is None:
            continue
        for k, record in sorted(records.items()):
            rows.append(
                {
                    "bb": bb,
                    "category": category,
                    "seed": seed,
                    "k": k,
                    **_score(record, images),
                }
            )
    return order_backbones(pd.DataFrame(rows))


def shots_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds of the category mean, per backbone and k; `population` is "all"
    or "without wood" (wood keeps a single defect type in the fixed population)."""
    parts = []
    for population, subset in (
        ("all", runs),
        ("without wood", runs[runs["category"] != "wood"]),
    ):
        summary = seed_summary(subset, ["bb", "k"], _VALUES)
        summary.insert(0, "population", population)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def shots_figure(table: pd.DataFrame):
    """The paired gain over kNN against k, one panel per backbone, with a band of one standard
    deviation over seeds. kNN itself does not depend on k here (same bank, same images), so its
    level is given in the panel title."""
    table = table[table["population"] == "all"]
    backbones = [str(b) for b in table["bb"].unique()]
    figure, axes = panels(len(backbones), sharey=True)
    for axis, bb in zip(axes, backbones, strict=True):
        rows = table[table["bb"] == bb].sort_values("k")
        x = np.arange(len(rows))
        mean, std = (
            rows["gain_auroc_mean"].to_numpy(),
            rows["gain_auroc_std"].to_numpy(),
        )
        axis.axhline(0, color=AXIS, linewidth=0.8)
        axis.fill_between(
            x, mean - std, mean + std, color=BACKBONE_COLOR[bb], alpha=0.18, linewidth=0
        )
        axis.plot(x, mean, color=BACKBONE_COLOR[bb], marker="o")
        axis.set_xticks(x, [str(k) for k in rows["k"]])
        axis.set_title(
            f"{BACKBONE_TITLE[bb]}  (kNN {rows['knn_auroc_mean'].iloc[0]:.2f})",
            color=SECONDARY,
            loc="left",
        )
        axis.set_xlabel("labelled faults per defect type")
    axes[0].set_ylabel("image AUROC gain over kNN (pt)")
    return figure


def write_shots(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    runs = shots_runs(study)
    table = shots_table(runs)
    header = ["backbone", "k", "kNN", "neg. bank", "gain", "recall gain"]
    rows = [
        [
            str(r["bb"]) if r["k"] == table["k"].min() else "",
            str(r["k"]),
            mean_std(r["knn_auroc_mean"], r["knn_auroc_std"], 2),
            mean_std(r["neg_auroc_mean"], r["neg_auroc_std"], 2),
            mean_std(r["gain_auroc_mean"], r["gain_auroc_std"], 2, signed=True),
            mean_std(r["gain_recall_mean"], r["gain_recall_std"], 1, signed=True),
        ]
        for r in table[table["population"] == "all"].to_dict("records")
    ]
    one = runs[
        (runs["bb"] == runs["bb"].iloc[0])
        & (runs["seed"] == runs["seed"].min())
        & (runs["k"] == runs["k"].min())
    ]
    note = (
        "Image AUROC in percent on a fixed set of test images (those left at the largest k, "
        f"defect types with support only): {one['category'].nunique()} categories, "
        f"{int(one['n_faulty'].sum())} faulty and {int(one['n_normal'].sum())} normal images per seed. "
        "Mean ± standard deviation over 5 seeds of the mean over categories; gains are paired. "
        "Recall is at the threshold set on the validation normals. Mask negatives, λ selected per run."
    )
    paths = write_table(out_dir, "shots", table, header, rows, "llrrrr", note)
    runs.to_csv(out_dir / "shots_per_run.csv", index=False)
    return [
        *paths,
        out_dir / "shots_per_run.csv",
        *save(shots_figure(table), out_dir, "shots"),
    ]
