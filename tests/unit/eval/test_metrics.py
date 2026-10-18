from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from cmad.eval import aupro, evaluate_category, image_metrics, pixel_auroc
from cmad.scorers import ScoreOutput


def test_image_metrics_match_sklearn_and_perfect_separation() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = rng.normal(size=200) + labels

    m = image_metrics(scores, labels)

    assert m["image_auroc"] == pytest.approx(roc_auc_score(labels, scores))
    assert m["image_ap"] == pytest.approx(average_precision_score(labels, scores))
    perfect = image_metrics(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert perfect["image_auroc"] == 1.0 and perfect["tpr_at_1pct_fpr"] == 1.0


def test_tpr_at_one_percent_fpr_on_a_hand_built_example() -> None:
    # 100 normals scored 0..99 (so exactly one normal is above 98), 4 anomalies: 3 above that
    scores = np.concatenate([np.arange(100.0), [99.5, 200, 201, 50]])
    labels = np.concatenate([np.zeros(100), np.ones(4)])

    assert image_metrics(scores, labels)["tpr_at_1pct_fpr"] == pytest.approx(0.75)


def test_pixel_auroc_perfect_and_random() -> None:
    masks = torch.zeros(4, 16, 16, dtype=torch.bool)
    masks[:, 4:8, 4:8] = True
    assert pixel_auroc(masks.float(), masks) == 1.0

    rand = torch.rand(4, 16, 16, generator=torch.Generator().manual_seed(0))
    assert 0.4 < pixel_auroc(rand, masks) < 0.6


def test_aupro_perfect_and_inverted() -> None:
    pytest.importorskip("anomalib.metrics")
    masks = torch.zeros(4, 32, 32, dtype=torch.bool)
    masks[:, 8:16, 8:16] = True
    masks[2:] = False  # two normal images as well
    perfect = masks.float()

    assert aupro(perfect, masks) == pytest.approx(1.0, abs=0.02)
    assert aupro(1.0 - perfect, masks) < 0.2


def test_evaluate_category_on_a_synthetic_category(tmp_path: Path) -> None:
    res = 32
    raw = tmp_path / "raw"
    (raw / "ds").mkdir(parents=True)
    m = np.zeros((res, res), dtype=np.uint8)
    m[8:16, 8:16] = 255
    Image.fromarray(m).save(raw / "ds" / "m0.png")
    rows = pd.DataFrame(
        {
            "image_id": ["g0", "g1", "b0"],
            "label": [0, 0, 1],
            "mask_path": [None, None, "ds/m0.png"],
        }
    )
    # a 4x4 patch map with a hot spot over the defect (rows 1-2, cols 1-2 of the grid)
    pm = torch.zeros(3, 4, 4)
    pm[2, 1:3, 1:3] = 1.0
    out = ScoreOutput(image_score=pm.flatten(1).max(1).values, patch_map=pm)

    r = evaluate_category(out, rows, raw, res=res, with_aupro=False)

    assert r["image_auroc"] == 1.0
    assert r["pixel_auroc"] > 0.95
    assert (r["n_test"], r["n_anomalous"]) == (3, 1)
    assert "aupro_0.3" not in r
    with pytest.raises(ValueError):
        evaluate_category(out, rows.iloc[:2], raw, res=res, with_aupro=False)


def test_threshold_metrics_on_a_hand_built_example() -> None:
    from cmad.eval import threshold_metrics

    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.4])
    labels = np.array([0, 0, 0, 1, 1, 1])

    m = threshold_metrics(scores, labels, tau=0.5)  # flagged: 0.9 (normal), 0.8 (anomaly)

    assert m["precision_at_tau_val"] == pytest.approx(0.5)
    assert m["recall_at_tau_val"] == pytest.approx(1 / 3)
    assert m["f1_at_tau_val"] == pytest.approx(0.4)
    assert m["fpr_at_tau_val"] == pytest.approx(1 / 3)
    assert threshold_metrics(scores, labels, tau=10.0)["f1_at_tau_val"] == 0.0  # nothing flagged
