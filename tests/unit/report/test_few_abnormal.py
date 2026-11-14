from pathlib import Path

import pandas as pd
import pytest

from cmad.report.few_abnormal import few_abnormal_table, paired_runs, write_few_abnormal
from cmad.report.runs import N_CATEGORIES, SEEDS

CATS = [f"cat{i:02d}" for i in range(N_CATEGORIES)]
METRICS = ["image_auroc", "aupro_0.3", "fpr_at_tau_val", "recall_at_tau_val"]


def study(gain: float = 1.0, inactive: tuple[str, ...] = (), ks=(1, 5)) -> pd.DataFrame:
    """kNN and negative-bank runs for one backbone: kNN scores 90 + seed on every metric, the
    negative bank `gain` more, except in the `inactive` categories, which fall back to kNN."""
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
                    "split_sha1": f"{category}-{k}-{seed}",
                }
                knn = {m: 90.0 + seed for m in METRICS}
                active = category not in inactive
                neg = {m: v + (gain if active else 0.0) for m, v in knn.items()}
                rows.append({**base, **knn, "scorer": "knn"})
                rows.append(
                    {
                        **base,
                        **neg,
                        "scorer": "negbank",
                        "source": "gt_mask",
                        "scale": "none",
                        "neg_active": active,
                        "selected_lambda": 0.5 if active else None,
                        "top_frac": None,
                        "bank_precision": 1.0 if active else None,
                        "bank_recall": 1.0 if active else None,
                    }
                )
    return pd.DataFrame(rows)


def test_gain_is_the_paired_difference() -> None:
    table = few_abnormal_table(study(gain=1.0))

    row = table[(table["k"] == 5) & (table["method"] == "gain")].iloc[0]
    assert row["image_auroc_mean"] == pytest.approx(1.0)
    assert row["image_auroc_std"] == pytest.approx(
        0.0
    )  # the seed effect cancels in the pair
    knn = table[(table["k"] == 5) & (table["method"] == "knn")].iloc[0]
    assert knn["image_auroc_mean"] == pytest.approx(92.0)
    assert knn["n_categories"] == N_CATEGORIES and knn["n_runs"] == N_CATEGORIES * len(
        SEEDS
    )


def test_categories_without_a_negative_bank_leave_both_rows() -> None:
    pairs = paired_runs(study(inactive=("cat00", "cat01"), ks=(10,)))

    assert pairs["category"].nunique() == N_CATEGORIES - 2
    assert (pairs["gain_image_auroc"] == 1.0).all()


def test_a_bank_active_in_some_seeds_only_is_refused() -> None:
    runs = study()
    one = (
        (runs["scorer"] == "negbank")
        & (runs["category"] == "cat00")
        & (runs["seed"] == 0)
    )
    runs.loc[one & (runs["k"] == 1), "neg_active"] = False

    with pytest.raises(ValueError, match="active in some seeds only"):
        paired_runs(runs)


def test_a_pair_on_different_splits_is_refused() -> None:
    runs = study()
    runs.loc[(runs["scorer"] == "knn") & (runs["seed"] == 0), "split_sha1"] = "other"

    with pytest.raises(ValueError, match="different splits"):
        paired_runs(runs)


def test_a_missing_run_stops_the_table() -> None:
    with pytest.raises(ValueError, match="few-abnormal kNN"):
        few_abnormal_table(study().iloc[2:])


def test_other_sources_and_normalised_runs_stay_out_of_the_table() -> None:
    runs = study()
    extra = runs[runs["scorer"] == "negbank"].copy()
    extra["image_auroc"] = 0.0
    top_q = extra.assign(source="top_q")
    normalised = extra.assign(scale="val_median")

    table = few_abnormal_table(pd.concat([runs, top_q, normalised], ignore_index=True))

    neg = table[table["method"] == "neg"]
    assert neg["image_auroc_mean"].tolist() == pytest.approx([93.0, 93.0])


def test_the_three_files_are_written(tmp_path: Path) -> None:
    paths = write_few_abnormal(study(gain=-0.5), tmp_path)

    assert [p.suffix for p in paths] == [".csv", ".md", ".tex"]
    assert "| S | 1 | 15 | kNN | 92.00 ± 1.58 |" in paths[1].read_text()
    assert "−0.50 ± 0.00" in paths[1].read_text()
    tex = paths[2].read_text()
    assert r"FPR at $\tau_{\mathrm{val}}$" in tex and r"$-$0.50 $\pm$ 0.00" in tex
    assert len(pd.read_csv(paths[0])) == 6  # 2 k x 3 methods
