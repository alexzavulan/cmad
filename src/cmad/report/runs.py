"""Run records as one table, and the rules that decide which records a result table may use.

The results directory keeps every record ever written: earlier settings, reruns after a split
changed, development runs. Each table therefore starts from `study_runs`, which keeps only the
records made with the study's settings, and checks its cells with `require_complete`, so that a
missing or doubled run stops the report instead of shifting a mean.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# The settings every study run of a backbone shares. The image score is the mean of the top 1 %
# of patch scores for DINOv2 and the maximum patch score for WRN-50.
STUDY = {
    "dinov2_vits14": {
        "label": "S",
        "layers": "last_raw",
        "metric": "cosine",
        "image_agg": "top1pct",
        "res": 448,
    },
    "dinov2_vitb14": {
        "label": "B",
        "layers": "last_raw",
        "metric": "cosine",
        "image_agg": "top1pct",
        "res": 448,
    },
    "wrn50_in1k": {
        "label": "WRN",
        "layers": "cnn_l23_pc_raw",
        "metric": "euclidean",
        "image_agg": "max",
        "res": 224,
    },
}
BACKBONE_ORDER = [s["label"] for s in STUDY.values()]
STUDY_SCORERS = ("knn", "negbank")
N_CATEGORIES = 15
SEEDS = (0, 1, 2, 3, 4)

# Everything that makes two runs of the same category, regime and seed different experiments
FACTORS = [
    "dataset",
    "backbone",
    "layers",
    "scorer",
    "metric",
    "coreset",
    "knn_k",
    "image_agg",
    "res",
    "resize",
    "source",
    "top_frac",
    "scale",
    "lam",
]
CELL = [*FACTORS, "category", "regime", "seed"]

# Stored as fractions, shown in percent. tau_val is a score threshold and keeps its unit.
PERCENT_METRICS = (
    "image_auroc",
    "image_ap",
    "pixel_auroc",
    "aupro_0.3",
    "fpr_at_tau_val",
    "recall_at_tau_val",
    "precision_at_tau_val",
    "f1_at_tau_val",
    "tpr_at_1pct_fpr",
)
_REGIME = re.compile(r"^(?P<family>.+?)_k(?P<k>\d+)$")


def threshold_metrics(
    val_scores: Sequence[float], test_scores: Sequence[float], labels: Sequence[int]
) -> dict[str, float]:
    """The operating point with the threshold at the largest validation-normal score: an image
    is flagged when its score is above it. Rates as fractions; precision and F1 are 0 when
    nothing is flagged."""
    tau = float(np.max(val_scores))
    flagged = np.asarray(test_scores) > tau
    faulty = np.asarray(labels) == 1
    hits = int((flagged & faulty).sum())
    recall = hits / int(faulty.sum())
    precision = hits / int(flagged.sum()) if flagged.any() else 0.0
    return {
        "tau_val": tau,
        "fpr_at_tau_val": float(flagged[~faulty].mean()),
        "recall_at_tau_val": recall,
        "precision_at_tau_val": precision,
        "f1_at_tau_val": 2 * precision * recall / (precision + recall) if hits else 0.0,
    }


def _flatten(record: dict, path: Path) -> dict:
    params = record.get("scorer_params", {})
    neg = record.get("neg_bank") or {}
    regime = record.get("regime", "")
    match = _REGIME.match(regime)
    row = {
        "path": str(path),
        "config_hash": record.get("config_hash"),
        "git_sha": (record.get("git_sha") or "")[:7],
        "timestamp": record.get("timestamp"),
        "dataset": record.get("dataset"),
        "category": record.get("category"),
        "regime": regime,
        # few_abn_k5 -> family few_abn, k 5: k counts labelled faults per defect type in the
        # few-abnormal regimes and training normals in the few-normal ones
        "family": match["family"] if match else regime,
        "k": int(match["k"]) if match else None,
        "seed": record.get("seed"),
        "split_sha1": record.get("split_sha1"),
        "backbone": record.get("backbone"),
        "layers": "+".join(record.get("layers", [])),
        "scorer": record.get("scorer"),
        "res": record.get("res"),
        "resize": record.get("resize"),
        "metric": params.get("metric"),
        "coreset": params.get("coreset_fraction"),
        "knn_k": params.get("k"),
        "image_agg": params.get("image_agg", "max"),
        "source": params.get("source"),
        "top_frac": params.get("top_frac"),
        "scale": params.get("scale", "none")
        if record.get("scorer") == "negbank"
        else None,
        "lam": None if params.get("lam") is None else str(params["lam"]),
        "has_image_scores": "image_scores" in record,
        "neg_active": neg.get("active"),
        "selected_lambda": neg.get("selected_lambda"),
        # share of the negative bank's patches that lie on a defect, and share of the defect
        # patches that the bank holds (both over all support images)
        "bank_precision": ((neg.get("selection") or {}).get("all") or {}).get(
            "precision"
        ),
        "bank_recall": ((neg.get("selection") or {}).get("all") or {}).get("recall"),
    }
    metrics = dict(record.get("metrics") or {})
    scores = record.get("image_scores") or {}
    if scores.get("val", {}).get("score") and scores.get("test", {}).get("score"):
        # Recomputed from the stored scores, so that every record follows the same threshold
        # rule whatever the code that wrote it did
        test = scores["test"]
        metrics.update(
            threshold_metrics(scores["val"]["score"], test["score"], test["label"])
        )
    for name, value in metrics.items():
        scale = 100.0 if name in PERCENT_METRICS else 1.0
        row[name] = None if value is None else scale * value
    return row


def load_records(runs_dir: Path) -> pd.DataFrame:
    """One row per result file. Metrics are in percent; `path` leads back to the full record
    (per-image scores, lambda curves), which is read only by the views that need it."""
    rows = [
        _flatten(json.loads(path.read_text()), path)
        for path in sorted(Path(runs_dir).glob("*.json"))
    ]
    frame = pd.DataFrame(rows)
    for column in ("k", "seed", "res"):
        if column in frame:
            frame[column] = frame[column].astype("Int64")
    return frame


def latest_per_cell(records: pd.DataFrame) -> pd.DataFrame:
    """Where the same experiment was run twice on a category, regime and seed (a rerun after the
    split changed), keep the newest record."""
    ordered = records.sort_values("timestamp")
    return ordered.drop_duplicates(CELL, keep="last").sort_index()


def study_runs(
    records: pd.DataFrame, layers: dict[str, str] | None = None
) -> pd.DataFrame:
    """The records made with the study's settings, newest per cell, with the short backbone name
    in `bb`. `layers` replaces a backbone's layer, e.g. {"dinov2_vits14": "8_raw"}."""
    layers = layers or {}
    keep = (
        (records["dataset"] == "mvtec_ad")
        & records["scorer"].isin(STUDY_SCORERS)
        & (records["coreset"] == 1.0)
        & (records["knn_k"] == 1)
        & records["resize"].isna()
        & records["has_image_scores"]
    )
    per_backbone = pd.Series(False, index=records.index)
    for backbone, settings in STUDY.items():
        per_backbone |= (
            (records["backbone"] == backbone)
            & (records["layers"] == layers.get(backbone, settings["layers"]))
            & (records["metric"] == settings["metric"])
            & (records["image_agg"] == settings["image_agg"])
            & (records["res"] == settings["res"])
        )
    study = latest_per_cell(records[keep & per_backbone]).copy()
    study["bb"] = pd.Categorical(
        study["backbone"].map({b: s["label"] for b, s in STUDY.items()}),
        categories=BACKBONE_ORDER,
        ordered=True,
    )
    return study


def order_backbones(frame: pd.DataFrame) -> pd.DataFrame:
    """Give `bb` its study order (S, B, WRN) back after it passed through plain values."""
    frame = frame.copy()
    frame["bb"] = pd.Categorical(
        frame["bb"].astype(str), categories=BACKBONE_ORDER, ordered=True
    )
    return frame


def require_complete(
    runs: pd.DataFrame,
    by: Sequence[str],
    n_categories: int = N_CATEGORIES,
    seeds: Iterable[int] = SEEDS,
    what: str = "runs",
) -> None:
    """Every group of `by` must hold each category once per seed: `n_categories` categories and
    exactly the given seeds. Anything else raises, naming the groups that are off."""
    seeds = sorted(seeds)
    if runs.empty:
        raise ValueError(f"{what}: no records")
    problems = []
    for key, group in runs.groupby(list(by), observed=True, dropna=False):
        per_category = group.groupby("category")["seed"].apply(lambda s: sorted(s))
        bad = [c for c, found in per_category.items() if found != seeds]
        if len(per_category) != n_categories or bad:
            problems.append(
                f"{key}: {len(group)} records, {len(per_category)} categories"
                + (f", wrong seeds in {bad}" if bad else "")
            )
    if problems:
        raise ValueError(
            f"{what}: expected {n_categories} categories x seeds {seeds} per group\n  "
            + "\n  ".join(problems)
        )


def seed_summary(
    runs: pd.DataFrame, by: Sequence[str], values: Sequence[str]
) -> pd.DataFrame:
    """Mean and standard deviation over seeds of the per-seed mean over categories.

    Returns one row per group of `by` with `<value>_mean`, `<value>_std` (sample standard
    deviation, n - 1) and `n_seeds`, `n_categories`, `n_runs`."""
    by = list(by)
    per_seed = runs.groupby([*by, "seed"], observed=True, dropna=False)[
        list(values)
    ].mean()
    grouped = per_seed.groupby(by, observed=True, dropna=False)
    out = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    counts = runs.groupby(by, observed=True, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        n_categories=("category", "nunique"),
        n_runs=("seed", "size"),
    )
    return out.join(counts).reset_index()
