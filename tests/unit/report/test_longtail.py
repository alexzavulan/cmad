"""The view of the long-tail runs."""

import itertools

import numpy as np
import pandas as pd
import pytest

from cmad.report.longtail import (
    at_budget,
    head_table,
    headroom_table,
    longtail_runs,
    ratio_table,
    score_run,
    type_changes,
    type_table,
)
from cmad.report.runs import SEEDS, order_backbones

LAMBDAS = [0.0, 1.0, 2.0]


def _record(selected: float = 2.0) -> dict:
    """Two normals; 'crack' (head, 3 labelled faults, 2 test faults) and 'hole' (tail, 1 labelled
    fault, 2 test faults), plus 'x', a hole image that is support at another ratio and so not in
    the common test set; its score would break the hole AUROC at lambda = 0."""
    ids = ["g0", "g1", "c0", "c1", "h0", "h1", "x"]
    curve = {
        "0": [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0],
        "1": [0.1, 0.4, 0.3, 0.6, 0.7, 0.8, 0.0],
        "2": [0.1, 0.4, 0.2, 0.3, 0.9, 0.9, 0.0],
    }
    return {
        "split": {
            "longtail": {
                "order": ["crack", "hole"],
                "counts": {"crack": 3, "hole": 1},
                "total": 4,
                "realised_ratio": 3.0,
            }
        },
        "neg_bank": {"lambdas": LAMBDAS, "selected_lambda": selected},
        "image_scores": {
            "test": {
                "image_id": ids,
                "label": [0, 0, 1, 1, 1, 1, 1],
                "defect_type": [
                    "good",
                    "good",
                    "crack",
                    "crack",
                    "hole",
                    "hole",
                    "hole",
                ],
                "curve": curve,
            },
            "val": {"curve": {"0": [0.45], "1": [0.45], "2": [0.45]}},
        },
    }


def _at(rows: list[dict], at: str, defect_type: str) -> dict:
    return next(r for r in rows if r["at"] == at and r["defect_type"] == defect_type)


def test_runs_are_scored_on_the_common_test_set_per_type_and_overall() -> None:
    rows = score_run(_record(), common={"g0", "g1", "c0", "c1", "h0", "h1"})

    hole = _at(rows, "knn", "hole")
    assert hole["n_faults"] == 2 and hole["image_auroc"] == pytest.approx(100.0)
    assert hole["count"] == 1 and not hole["head"]
    crack = _at(rows, "fixed", "crack")
    assert crack["head"] and crack["count"] == 3
    # at lambda = 1, crack c0 (0.3) falls below normal g1 (0.4): 3 of 4 pairs
    assert crack["image_auroc"] == pytest.approx(75.0)
    assert crack["recall"] == 50.0  # threshold 0.45: only c1
    every = _at(rows, "selected", "all")
    assert every["lam"] == 2.0 and every["n_faults"] == 4 and every["count"] == 4
    assert every["fpr"] == 0.0


def test_a_type_without_common_test_images_is_an_error() -> None:
    with pytest.raises(ValueError, match="no common test image of type 'hole'"):
        score_run(_record(), common={"g0", "g1", "c0", "c1"})


def _study(drop=None) -> pd.DataFrame:
    rows = [
        {
            "bb": "S",
            "category": f"c{c}",
            "regime": f"longtail_r{rho}",
            "scorer": "negbank",
            "scale": "val_median",
            "seed": seed,
            "path": "",
        }
        for c, rho, seed in itertools.product(range(14), (1, 3, 10), SEEDS)
        if (c, rho, seed) != drop
    ]
    return order_backbones(pd.DataFrame(rows))


def test_every_category_needs_every_ratio_and_seed() -> None:
    assert len(longtail_runs(_study())) == 14 * 3 * 5
    with pytest.raises(ValueError, match="wrong cells in \\['c3'\\]"):
        longtail_runs(_study(drop=(3, 10, 2)))


def _scores(cells: list[dict]) -> pd.DataFrame:
    """Score rows from compact cells: category, seed, rho, at, defect type, AUROC."""
    base = {
        "bb": "S",
        "recall": 90.0,
        "fpr": 5.0,
        "budget": 10,
        "realised_ratio": 1.0,
        "selected_lambda": 0.0,
        "head_share": 0.5,
        "count": 5,
        "head": False,
    }
    return order_backbones(pd.DataFrame([{**base, **c} for c in cells]))


def test_the_gain_at_each_ratio_is_paired_with_the_same_category_and_seed() -> None:
    cells = []
    for category, shift in (("a", 0.0), ("b", 1.0)):
        for rho, gain in ((1, 1.0), (3, 1.0), (10, 0.5)):
            total = (
                10 if category == "a" else 8
            )  # b loses budget to the cap at rho = 10
            common = {"category": category, "seed": 0, "rho": rho, "total": total}
            cells += [
                {
                    **common,
                    "at": "knn",
                    "defect_type": "all",
                    "image_auroc": 90 + shift,
                },
                {
                    **common,
                    "at": "fixed",
                    "defect_type": "all",
                    "image_auroc": 90 + shift + gain,
                },
                {
                    **common,
                    "at": "selected",
                    "defect_type": "all",
                    "image_auroc": 90 + shift,
                },
            ]
    scores = _scores(cells)

    table = ratio_table(scores).set_index("rho")

    assert table.loc["1", "fixed"] == pytest.approx(1.0)
    assert table.loc["10", "fixed"] == pytest.approx(0.5)
    assert table.loc["10", "vs_rho1"] == pytest.approx(-0.5)
    assert table.loc["3", "image_auroc"] == pytest.approx(90.5)  # kNN, both categories
    assert at_budget(scores) == {"a"}
    assert table.loc["10, at budget", "n_categories"] == 1


def test_a_type_is_compared_with_itself_at_rho_1_and_ceilings_are_marked() -> None:
    cells = []
    for rho, count, crack, hole in ((1, 5, 100.0, 95.0), (10, 9, 100.0, 94.0)):
        common = {"category": "a", "seed": 0, "rho": rho, "at": "fixed", "total": 10}
        cells += [
            {**common, "defect_type": "crack", "count": count, "image_auroc": crack},
            {**common, "defect_type": "hole", "count": 10 - count, "image_auroc": hole},
        ]
    changes = type_changes(_scores(cells)).set_index("defect_type")

    assert changes.loc["hole", "d_auroc"] == pytest.approx(-1.0)
    assert changes.loc["hole", "count"] == 1 and not changes.loc["hole", "ceiling"]
    assert changes.loc["crack", "d_auroc"] == 0.0 and changes.loc["crack", "ceiling"]
    table = type_table(changes.reset_index()).set_index("bin")
    assert table.loc["1", "d_auroc"] == pytest.approx(-1.0)
    assert table.loc["≥ 9", "ceiling"] == 100.0


def test_the_head_table_splits_head_and_tail_types() -> None:
    cells = []
    for at, head_auroc, tail_auroc in (("fixed", 99.0, 97.0), ("selected", 99.5, 96.0)):
        common = {
            "category": "a",
            "seed": 0,
            "rho": 10,
            "at": at,
            "total": 13,
            "head_share": 12 / 13,
            "selected_lambda": 0.0,
        }
        cells += [
            {
                **common,
                "defect_type": "crack",
                "count": 12,
                "head": True,
                "image_auroc": head_auroc,
            },
            {**common, "defect_type": "hole", "count": 1, "image_auroc": tail_auroc},
        ]
    row = head_table(_scores(cells)).iloc[0]

    assert row["head"] == pytest.approx(0.5) and row["tail"] == pytest.approx(-1.0)
    assert row["lambda_0"] == 100.0 and row["head_share"] == pytest.approx(12 / 13)


def test_the_headroom_table_keeps_types_knn_does_not_separate_and_pairs_with_rho_1() -> (
    None
):
    cells = []
    for seed, gain in ((0, 1.0), (1, 3.0)):
        for rho, count, change in ((1, 5, 0.0), (10, 1, -0.5)):
            common = {"category": "a", "seed": seed, "rho": rho, "total": 10}
            for defect_type, knn in (("hole", 95.0), ("crack", 100.0)):
                extra = gain + change if defect_type == "hole" else 0.0
                cells += [
                    {
                        **common,
                        "at": "knn",
                        "defect_type": defect_type,
                        "count": count,
                        "image_auroc": knn,
                    },
                    {
                        **common,
                        "at": "fixed",
                        "defect_type": defect_type,
                        "count": count,
                        "image_auroc": knn + extra,
                    },
                ]
    table = headroom_table(_scores(cells)).set_index("bin")

    assert list(table.index) == ["5 (ρ = 1)", "1"]  # crack is at 100 and left out
    assert table.loc["5 (ρ = 1)", "gain"] == pytest.approx(2.0)
    # seeds are averaged within the one category, so there is no spread to report
    assert np.isnan(table.loc["5 (ρ = 1)", "gain_se"])
    assert table.loc["1", "change"] == pytest.approx(-0.5)
    assert table.loc["1", "n"] == 2 and table.loc["1", "n_categories"] == 1
