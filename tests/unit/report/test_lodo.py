"""The view of the runs with a held-out defect type."""

import numpy as np
import pandas as pd
import pytest

from cmad.report.lodo import _paired, _summary, lodo_runs, lodo_table, score_run
from cmad.report.runs import order_backbones

LAMBDAS = [0.0, 1.0, 2.0]


def _record(held_out: str = "hole", selected: float = 1.0) -> dict:
    """Two normals; seen type 'crack' (2 faults), held-out 'hole' (2), too-small 'tiny' (1).
    At lambda = 1 the seen faults rise and one held-out fault falls below a normal image."""
    labels = [0, 0, 1, 1, 1, 1, 1]
    types = ["good", "good", "crack", "crack", "hole", "hole", "tiny"]
    seen = [None, None, True, True, False, False, False]
    curve = {
        "0": [0.1, 0.4, 0.3, 0.6, 0.7, 0.8, 0.5],
        "1": [0.1, 0.4, 0.7, 0.8, 0.3, 0.8, 0.2],
        "2": [0.1, 0.4, 0.9, 0.9, 0.0, 0.0, 0.0],
    }
    return {
        "split": {"held_out_type": held_out},
        "neg_bank": {"lambdas": LAMBDAS, "selected_lambda": selected},
        "image_scores": {
            "test": {
                "label": labels,
                "defect_type": types,
                "seen": seen,
                "curve": curve,
            },
            "val": {"curve": {"0": [0.45], "1": [0.45], "2": [0.45]}},
        },
    }


def _at(rows: list[dict], lam: float, subset: str) -> dict:
    return next(r for r in rows if r["lam"] == lam and r["subset"] == subset)


def test_the_unseen_subset_is_the_held_out_type_only() -> None:
    rows = score_run(_record())

    unseen = _at(rows, 0.0, "unseen")
    assert (
        unseen["n_faults"] == 2
    )  # 'tiny' is flagged unseen too, but is not the held-out type
    assert unseen["image_auroc"] == pytest.approx(100.0)
    # at lambda = 1 the held-out fault at 0.3 is below the normal at 0.4: 3 of 4 pairs right
    assert _at(rows, 1.0, "unseen")["image_auroc"] == pytest.approx(75.0)
    assert _at(rows, 0.0, "seen")["image_auroc"] == pytest.approx(75.0)
    assert _at(rows, 1.0, "seen")["image_auroc"] == pytest.approx(100.0)
    assert _at(rows, 0.0, "too_small")["n_faults"] == 1
    # threshold 0.45 at every lambda: recall and the false-positive rate follow it
    assert (
        _at(rows, 1.0, "unseen")["recall"] == 50.0
        and _at(rows, 1.0, "seen")["recall"] == 100.0
    )
    assert _at(rows, 0.0, "seen")["fpr"] == 0.0
    assert [r["selected"] for r in rows if r["subset"] == "seen"] == [
        False,
        True,
        False,
    ]


def _curves(diffs: dict[tuple[str, str], list[float]], marked=()) -> pd.DataFrame:
    """Unseen-subset rows at lambda 0 (AUROC 90) and 1 (90 + diff), one per seed in the list."""
    rows = []
    for (category, held_out), per_seed in diffs.items():
        for seed, diff in enumerate(per_seed):
            base = {
                "bb": "S",
                "category": category,
                "held_out": held_out,
                "seed": seed,
                "subset": "unseen",
                "marked": held_out in marked,
                "recall": 80.0,
                "fpr": 5.0,
                "n_faults": 3,
            }
            rows.append({**base, "lam": 0.0, "selected": False, "image_auroc": 90.0})
            rows.append(
                {**base, "lam": 1.0, "selected": True, "image_auroc": 90.0 + diff}
            )
    return order_backbones(pd.DataFrame(rows))


def test_units_average_their_seeds_and_the_two_standard_errors_differ() -> None:
    curves = _curves(
        {("a", "x"): [-1.0, -3.0], ("a", "y"): [-1.0, -1.0], ("b", "z"): [1.0, 1.0]}
    )

    s = _summary(_paired(curves, "unseen", "fixed"))

    assert s["n_units"] == 3 and s["n_categories"] == 2
    assert s["mean"] == pytest.approx((-2.0 - 1.0 + 1.0) / 3)
    assert s["se_types"] == pytest.approx(
        np.std([-2.0, -1.0, 1.0], ddof=1) / np.sqrt(3)
    )
    # categories: a = -1.5, b = 1
    assert s["se_categories"] == pytest.approx(np.std([-1.5, 1.0], ddof=1) / np.sqrt(2))
    assert s["n_below"] == 2 and s["worst"] == -2.0 and s["worst_unit"] == "a / x"


def test_the_rule_reads_the_unseen_subset_at_lambda_1_without_the_marked_folds() -> (
    None
):
    clearly_below = {("a", f"t{i}"): [-1.0 - 0.1 * i] * 2 for i in range(4)}
    curves = _curves({**clearly_below, ("b", "small"): [50.0, 50.0]}, marked=("small",))
    seen = curves.assign(subset="seen", image_auroc=90.0)

    table = lodo_table(pd.concat([curves, seen], ignore_index=True)).set_index(
        ["subset", "compare"]
    )

    rule = table.loc[("unseen", "fixed")]
    assert rule["n_units"] == 4  # the marked fold, far above, is not a unit
    assert rule["verdict"] == "below kNN: conditional"
    assert table.loc[("seen", "fixed"), "verdict"] == ""

    near_zero = _curves(
        {("a", "x"): [-0.2, 0.0], ("a", "y"): [0.1, 0.1], ("b", "z"): [0.0, -0.1]}
    )
    table = lodo_table(
        pd.concat([near_zero, near_zero.assign(subset="seen")])
    ).set_index(["subset", "compare"])
    assert table.loc[("unseen", "fixed"), "verdict"] == "not below kNN"


def test_an_incomplete_fold_stops_the_view() -> None:
    rows = [
        {
            "bb": "S",
            "family": "lodo",
            "scorer": "negbank",
            "scale": "val_median",
            "category": "a",
            "held_out": "x",
            "seed": s,
        }
        for s in range(4)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        lodo_runs(order_backbones(pd.DataFrame(rows)))
