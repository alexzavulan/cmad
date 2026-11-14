"""The views built on the loader: negative sources, normal-only tables, the curve over the
number of labelled faults, and the rules for choosing lambda."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmad.report.lambda_curves import (
    _best_elsewhere,
    lambda_runs,
    rule_runs,
    rule_table,
)
from cmad.report.neg_source import FRACTIONS, neg_source_table
from cmad.report.normal_only import few_normal_table, one_class_table, write_normal_only
from cmad.report.runs import N_CATEGORIES, SEEDS, threshold_metrics
from cmad.report.shots import fixed_population, shots_runs, shots_table, write_shots
from cmad.report.tables import mean_std

CATS = [f"cat{i:02d}" for i in range(N_CATEGORIES)]
METRICS = ["image_auroc", "aupro_0.3", "fpr_at_tau_val", "recall_at_tau_val"]


def test_threshold_is_the_largest_validation_score_and_flags_strictly_above() -> None:
    out = threshold_metrics([0.1, 0.5], [0.5, 0.6, 0.4, 0.9], [0, 0, 1, 1])

    assert out["tau_val"] == 0.5
    assert out["fpr_at_tau_val"] == 0.5  # 0.5 itself is not flagged, 0.6 is
    assert out["recall_at_tau_val"] == 0.5
    assert out["precision_at_tau_val"] == 0.5 and out["f1_at_tau_val"] == 0.5


def test_nothing_flagged_gives_zero_precision_and_f1() -> None:
    out = threshold_metrics([2.0], [0.5, 0.9], [0, 1])

    assert (
        out["recall_at_tau_val"],
        out["precision_at_tau_val"],
        out["f1_at_tau_val"],
    ) == (0, 0, 0)


def test_a_mean_that_rounds_to_zero_has_no_minus_sign() -> None:
    assert mean_std(-0.001, 0.05, 2, signed=True) == "+0.00 ± 0.05"
    assert mean_std(-0.3, 0.05, 1) == "−0.3 ± 0.1"


# ---------------------------------------------------------------- tables from the flat records


def _few_abnormal(
    sources: dict[tuple[str, float | None], float], ks=(1,)
) -> pd.DataFrame:
    """kNN at 90 on every metric and one negative-bank run set per source at 90 + its gain."""
    rows = []
    for k in ks:
        for category in CATS:
            for seed in SEEDS:
                base = {
                    "bb": "S",
                    "family": "few_abn",
                    "k": k,
                    "category": category,
                    "seed": seed,
                    "split_sha1": "s",
                }
                rows.append({**base, **dict.fromkeys(METRICS, 90.0), "scorer": "knn"})
                for (source, top_frac), gain in sources.items():
                    rows.append(
                        {
                            **base,
                            **dict.fromkeys(METRICS, 90.0 + gain),
                            "scorer": "negbank",
                            "source": source,
                            "top_frac": top_frac,
                            "scale": "none",
                            "neg_active": True,
                            "selected_lambda": 0.5,
                            "bank_precision": 1.0 if source == "gt_mask" else 0.4,
                            "bank_recall": 1.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_each_source_is_compared_with_knn_and_with_the_masks() -> None:
    sources = {("gt_mask", None): 2.0}
    sources |= {("top_q", q): 1.0 for q in FRACTIONS} | {
        ("random", q): 0.0 for q in FRACTIONS
    }

    table = neg_source_table(_few_abnormal(sources)).set_index("source")

    assert list(table.index[:2]) == ["masks", "farthest 1 %"]
    assert table.loc["farthest 10 %", "gain_mean"] == pytest.approx(1.0)
    assert table.loc["farthest 10 %", "vs_masks_mean"] == pytest.approx(-1.0)
    assert table.loc["random 25 %", "vs_masks_mean"] == pytest.approx(-2.0)
    assert table.loc["farthest 10 %", "bank_precision_mean"] == pytest.approx(40.0)


def test_a_missing_source_stops_the_table() -> None:
    with pytest.raises(ValueError, match="top_q"):
        neg_source_table(_few_abnormal({("gt_mask", None): 2.0}))


def _normal_only() -> pd.DataFrame:
    rows = []
    for bb, layers in (("S", "last_raw"), ("WRN", "cnn_l23_pc_raw")):
        for category in CATS:
            for seed in SEEDS:
                base = {
                    "bb": bb,
                    "layers": layers,
                    "scorer": "knn",
                    "category": category,
                    "seed": seed,
                }
                values = dict.fromkeys(
                    [
                        "image_auroc",
                        "image_ap",
                        "pixel_auroc",
                        "aupro_0.3",
                        "fpr_at_tau_val",
                        "recall_at_tau_val",
                        "f1_at_tau_val",
                    ],
                    99.0,
                )
                rows.append({**base, **values, "family": "oneclass", "k": pd.NA})
                for k in (1, 4):
                    rows.append(
                        {
                            **base,
                            **values,
                            "family": "few_norm",
                            "k": k,
                            "image_auroc": 80.0 + k,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["bb"] = pd.Categorical(
        frame["bb"], categories=["S", "B", "WRN"], ordered=True
    )
    frame["k"] = frame["k"].astype("Int64")
    return frame


def test_one_class_table_adds_the_other_layers_as_rows() -> None:
    study = _normal_only()
    middle = study[(study["bb"] == "S") & (study["family"] == "oneclass")].assign(
        layers="8_raw", image_auroc=98.0
    )

    table = one_class_table(study, middle)

    assert list(zip(table["bb"], table["layers"], strict=True)) == [
        ("S", "last_raw"),
        ("S", "8_raw"),
        ("WRN", "cnn_l23_pc_raw"),
    ]
    assert table["image_auroc_mean"].tolist() == pytest.approx([99.0, 98.0, 99.0])


def test_few_normal_table_ends_with_the_full_training_set(tmp_path: Path) -> None:
    study = _normal_only()

    table = few_normal_table(study)
    s = table[table["bb"] == "S"]

    assert s["image_auroc_mean"].tolist() == pytest.approx([81.0, 84.0, 99.0])
    assert pd.isna(s["k"].iloc[-1])
    paths = write_normal_only(study, None, tmp_path)
    text = (tmp_path / "few_normal_image_auroc.md").read_text()
    assert "| backbone | k = 1 | k = 4 | all |" in text
    assert "| WRN | 81.00 ± 0.00 | 84.00 ± 0.00 | 99.00 ± 0.00 |" in text
    assert all(p.exists() for p in paths)


# ------------------------------------------------------------------ views that read the scores

LAMBDAS = [0.0, 0.5, 1.0]


def _record(k: int, active: bool = True, selected: float = 0.5) -> dict:
    """Two normal test images and faults of two types; type 'rare' has no support at k = 10.
    At lambda = 0 one 'common' fault scores below a normal image; at 0.5 the order is right."""
    ids = ["n0", "n1", "c0", "c1", "r0"] + (["c2"] if k < 10 else [])
    labels = [0, 0, 1, 1, 1] + ([1] if k < 10 else [])
    types = ["good", "good", "common", "common", "rare"] + (
        ["common"] if k < 10 else []
    )
    at_zero = [0.1, 0.5, 0.4, 0.9, 0.05] + ([0.9] if k < 10 else [])
    curve = {
        "0": at_zero,
        "0.5": [0.1, 0.3, 0.6, 0.9, 0.05] + ([0.9] if k < 10 else []),
        "1": [0.9, 0.9, 0.1, 0.1, 0.05] + ([0.1] if k < 10 else []),
    }
    key = f"{selected:g}"
    return {
        "category": "cat",
        "regime": f"few_abn_k{k}",
        "neg_bank": {
            "active": active,
            "lambdas": LAMBDAS,
            "selected_lambda": selected if active else None,
            "support": {"s0": {"defect_type": "common"}} if active else {},
            "curve": {
                "0": {"image_auroc": 0.8},
                "0.5": {"image_auroc": 1.0},
                "1": {"image_auroc": 0.0},
            },
            "selection_rule": {"curve": {"0": 0.7, "0.5": 0.9, "1": 0.2}},
        },
        "image_scores": {
            "test": {
                "image_id": ids,
                "label": labels,
                "defect_type": types,
                "score": curve[key],
                "curve": curve,
            },
            "val": {
                "score": [0.35],
                "curve": {"0": [0.45], "0.5": [0.35], "1": [0.95]},
            },
        },
    }


def _study_with_files(
    tmp: Path, ks=(1, 10), inactive: tuple[str, ...] = ()
) -> pd.DataFrame:
    rows = []
    for k in ks:
        for category in CATS:
            for seed in SEEDS:
                active = not (category in inactive and k == 10)
                path = tmp / f"{category}-{k}-{seed}.json"
                path.write_text(json.dumps(_record(k, active)))
                rows.append(
                    {
                        "bb": "S",
                        "family": "few_abn",
                        "scorer": "negbank",
                        "source": "gt_mask",
                        "scale": "none",
                        "top_frac": np.nan,
                        "k": k,
                        "category": category,
                        "seed": seed,
                        "neg_active": active,
                        "path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def test_fixed_population_keeps_normals_and_the_supported_defect_types() -> None:
    assert fixed_population(_record(10)) == {"n0", "n1", "c0", "c1"}
    assert fixed_population(_record(10, active=False)) is None


def test_every_k_is_scored_on_the_images_left_at_the_largest_k(tmp_path: Path) -> None:
    runs = shots_runs(_study_with_files(tmp_path, inactive=("cat00",)))

    assert "cat00" not in set(
        runs["category"]
    )  # no bank at the largest k: not on the curve
    assert runs["category"].nunique() == N_CATEGORIES - 1
    row = runs[runs["k"] == 1].iloc[0]
    # 'c2' and the unsupported 'r0' are in the k = 1 test set but not in the fixed population
    assert (row["n_faulty"], row["n_normal"]) == (2, 2)
    assert row["knn_auroc"] == pytest.approx(
        75.0
    )  # one of four normal-fault pairs is inverted
    assert row["neg_auroc"] == pytest.approx(100.0) and row[
        "gain_auroc"
    ] == pytest.approx(25.0)
    # thresholds: 0.45 at lambda = 0 flags one of the two faults, 0.35 at the selected lambda both
    assert (row["knn_recall"], row["neg_recall"]) == (50.0, 100.0)

    table = shots_table(runs)
    assert set(table["population"]) == {"all", "without wood"}
    assert table["gain_auroc_mean"].tolist() == pytest.approx([25.0] * 4)


def test_shots_outputs_include_the_figure(tmp_path: Path) -> None:
    paths = write_shots(_study_with_files(tmp_path), tmp_path / "out")

    assert {p.suffix for p in paths} == {".csv", ".md", ".tex", ".pdf", ".png"}
    assert all(p.stat().st_size > 0 for p in paths)
    assert (
        "15 categories, 30 faulty and 30 normal"
        in (tmp_path / "out" / "shots.md").read_text()
    )


def test_lambda_runs_reset_the_threshold_at_each_lambda(tmp_path: Path) -> None:
    curves = lambda_runs(_study_with_files(tmp_path, ks=(1,)), "gt_mask", "none")

    one = curves[(curves["category"] == "cat00") & (curves["seed"] == 0)].set_index(
        "lam"
    )
    assert one["image_auroc"].tolist() == [80.0, 100.0, 0.0]
    assert (
        one.loc[0.0, "fpr"] == 50.0
    )  # threshold 0.45: the normal image at 0.5 is flagged
    assert one.loc[0.5, "fpr"] == 0.0 and one.loc[0.5, "recall"] == 75.0
    assert one["selected"].tolist() == [False, True, False]
    assert one.loc[0.5, "fold"] == 0.9


def _curves(
    folds: dict[str, list[float]],
    test: dict[str, list[float]],
    selected: dict[str, float],
) -> pd.DataFrame:
    """One backbone, seed and k; per category a fold curve and a test curve over LAMBDAS."""
    rows = []
    for category in folds:
        for i, lam in enumerate(LAMBDAS):
            rows.append(
                {
                    "bb": "S",
                    "k": 1,
                    "category": category,
                    "seed": 0,
                    "lam": lam,
                    "image_auroc": test[category][i],
                    "fpr": 5.0,
                    "recall": 90.0,
                    "fold": folds[category][i],
                    "selected": lam == selected[category],
                }
            )
    return pd.DataFrame(rows)


def test_shared_lambda_comes_from_the_other_categories_only() -> None:
    # a's own folds say 1.0, the other two say 0.5; a tie between lambdas goes to the smaller one
    folds = {
        "a": [0.0, 0.0, 1.0],
        "b": [0.5, 0.9, 0.1],
        "c": [0.5, 0.9, 0.1],
        "d": [np.nan] * 3,
    }
    test = {
        "a": [90.0, 95.0, 50.0],
        "b": [90.0, 91.0, 92.0],
        "c": [90.0, 91.0, 92.0],
        "d": [90.0, 80.0, 70.0],
    }
    curves = _curves(folds, test, selected={"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.0})

    assert _best_elsewhere(curves, "a") == 0.5
    assert _best_elsewhere(curves[curves["category"].isin(["a", "d"])], "a") is None
    tie = _curves(
        {"a": [0.9, 0.9, 0.1], "b": [0.9, 0.9, 0.1]}, test, {"a": 0.0, "b": 0.0}
    )
    assert _best_elsewhere(tie, "a") == 0.0

    by_rule = rule_runs(curves, fixed=1.0).set_index(["category", "rule"])
    assert by_rule.loc[("a", "selected"), "image_auroc"] == 50.0
    assert by_rule.loc[("a", "shared"), "image_auroc"] == 95.0
    assert by_rule.loc[("a", "shared_k"), "lam"] == 0.5  # k = 1 is one of the pooled k
    assert by_rule.loc[("d", "fixed_1"), "vs_knn"] == -20.0
    assert by_rule.loc[("d", "knn"), "vs_knn"] == 0.0

    table = rule_table(by_rule.reset_index()).set_index("rule")
    assert (
        table.loc["fixed_1", "n_harmed"] == 2 and table.loc["fixed_1", "worst"] == -40.0
    )
    assert table.loc["knn", "n_harmed"] == 0


def test_no_fixed_rule_unless_asked_for() -> None:
    curves = _curves(
        {"a": [0.5, 0.9, 0.1], "b": [0.5, 0.9, 0.1]},
        {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]},
        {"a": 0.0, "b": 0.0},
    )

    assert "fixed_1" not in set(rule_runs(curves)["rule"])
