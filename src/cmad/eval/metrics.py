from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from cmad.scorers import ScoreOutput

from .maps import upsample_and_blur
from .masks import load_masks


def image_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "tpr_at_1pct_fpr": float(tpr[fpr <= 0.01].max()),
    }


def pixel_auroc(maps: torch.Tensor, masks: torch.Tensor) -> float:
    return float(roc_auc_score(masks.flatten().numpy(), maps.flatten().numpy()))


def aupro(maps: torch.Tensor, masks: torch.Tensor, fpr_limit: float = 0.3) -> float:
    """Per-region overlap AUC up to `fpr_limit`, via anomalib (installed on the cluster)."""
    try:
        from anomalib.metrics import AUPRO
    except ImportError as err:
        raise ImportError(
            "AUPRO needs anomalib, which is only installed on the cluster"
        ) from err
    # anomalib's metric reads named fields off a batch object; here a plain namespace stands in
    metric = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=fpr_limit)
    metric.update(SimpleNamespace(anomaly_map=maps.float(), gt_mask=masks.int()))
    return float(metric.compute())


def evaluate_category(
    out: ScoreOutput,
    rows: pd.DataFrame,
    raw_root: Path,
    res: int = 224,
    resize: int | None = None,
    sigma: float = 4.0,
    with_aupro: bool = True,
    fpr_limit: float = 0.3,
) -> dict[str, float]:
    """Metrics (fractions in [0, 1]) for one category; `rows` must be in the scorer's image order."""
    if len(rows) != len(out.image_score):
        raise ValueError(
            f"{len(rows)} manifest rows for {len(out.image_score)} scored images"
        )
    labels = rows["label"].to_numpy()
    maps = upsample_and_blur(out.patch_map, size=res, sigma=sigma)
    masks = load_masks(rows, raw_root, res, resize)

    result = image_metrics(out.image_score.numpy(), labels)
    result["pixel_auroc"] = pixel_auroc(maps, masks)
    if with_aupro:
        result[f"aupro_{fpr_limit:g}"] = aupro(maps, masks, fpr_limit)
    result["n_test"] = len(rows)
    result["n_anomalous"] = int(labels.sum())
    return result
