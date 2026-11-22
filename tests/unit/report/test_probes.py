"""The linear-probe view."""

import itertools

import pandas as pd
import pytest

from cmad.report.probes import probe_few_runs, score_probe_run
from cmad.report.runs import SEEDS, order_backbones


def test_a_probe_run_is_scored_on_unseen_and_seen_faults_at_its_own_threshold() -> None:
    record = {
        "split": {"held_out_type": "hole"},
        "image_scores": {
            "test": {
                "label": [0, 0, 1, 1, 1],
                "defect_type": ["good", "good", "hole", "crack", "crack"],
                "seen": [None, None, False, True, True],
                "score": [0.0, 2.0, 1.0, 3.0, 4.0],
            },
            "val": {"score": [1.5]},
        },
    }
    rows = {r["subset"]: r for r in score_probe_run(record)}

    assert rows["unseen"]["image_auroc"] == pytest.approx(50.0)  # 1.0 between 0 and 2
    assert rows["unseen"]["recall"] == 0.0 and rows["seen"]["recall"] == 100.0
    assert rows["seen"]["image_auroc"] == pytest.approx(100.0)
    assert rows["seen"]["fpr"] == rows["unseen"]["fpr"] == 50.0


def _study(drop=None) -> pd.DataFrame:
    rows = [
        {
            "bb": "S",
            "family": "few_abn",
            "scorer": "probe",
            "loss": "wce",
            "probe_c": 1.0,
            "category": f"c{c}",
            "k": k,
            "seed": seed,
        }
        for c, k, seed in itertools.product(range(15), (1, 5, 10), SEEDS)
        if (c, k, seed) != drop
    ]
    return order_backbones(pd.DataFrame(rows))


def test_probe_runs_must_be_complete_per_backbone_and_k() -> None:
    assert len(probe_few_runs(_study(), "wce")) == 15 * 3 * 5
    with pytest.raises(ValueError, match="few-abnormal probe"):
        probe_few_runs(_study(drop=(4, 5, 0)), "wce")
