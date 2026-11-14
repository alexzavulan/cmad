import pandas as pd
import pytest
from scipy.stats import wilcoxon

from cmad.report.claims import _per_category, signed_rank


def test_zeros_are_dropped_and_the_signs_counted() -> None:
    d = pd.Series([0.0, 0.0, 0.5, 1.0, -0.2, 0.3, 0.8, 0.4])

    out = signed_rank(d)

    assert (out["n_units"], out["n_left"], out["n_positive"], out["n_negative"]) == (
        8,
        6,
        5,
        1,
    )
    assert out["mean"] == pytest.approx(d.mean())  # the mean keeps the zero categories
    assert out["p"] == pytest.approx(wilcoxon([0.5, 1.0, -0.2, 0.3, 0.8, 0.4]).pvalue)


def test_no_p_value_with_five_or_fewer_categories_left() -> None:
    out = signed_rank(pd.Series([0.0] * 10 + [1.0, 2.0, 3.0, 4.0, 5.0]))

    assert out["n_left"] == 5 and out["p"] is None
    # the reason: even five positive differences give no two-sided p below 0.05
    assert wilcoxon([1.0, 2.0, 3.0, 4.0, 5.0]).pvalue > 0.05


def test_held_out_types_are_averaged_within_a_category_before_categories() -> None:
    frame = pd.DataFrame(
        {
            "category": ["a", "a", "a", "a", "b", "b"],
            "held_out": ["x", "x", "y", "y", "z", "z"],
            "seed": [0, 1, 0, 1, 0, 1],
            "diff": [1.0, 3.0, 0.0, 0.0, 5.0, 5.0],
        }
    )

    per_category = _per_category(frame, "diff", ["seed", "held_out"])

    # a: types x (mean 2) and y (0) weigh the same, whatever their number of runs
    assert per_category.to_dict() == {"a": 1.0, "b": 5.0}
