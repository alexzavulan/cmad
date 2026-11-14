import json
from pathlib import Path

import pandas as pd
import pytest

from cmad.report.runs import load_records, require_complete, seed_summary, study_runs

CATS = ["bottle", "cable", "screw"]


def write_record(tmp: Path, name: str, **changes) -> None:
    """A DINOv2-S study run on few_abn_k5; `changes` replace top-level or scorer_params fields."""
    record = {
        "config_hash": name,
        "git_sha": "abcdef0123",
        "timestamp": "2026-11-01T00:00:00+00:00",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "regime": "few_abn_k5",
        "seed": 0,
        "split_sha1": "s0",
        "backbone": "dinov2_vits14",
        "layers": ["last_raw"],
        "scorer": "knn",
        "res": 448,
        "resize": None,
        "scorer_params": {
            "metric": "cosine",
            "coreset_fraction": 1.0,
            "k": 1,
            "image_agg": "top1pct",
        },
        "metrics": {"image_auroc": 0.99, "aupro_0.3": 0.9, "tau_val": 0.2},
        "image_scores": {},
    }
    params = {
        k: changes.pop(k)
        for k in list(changes)
        if k in ("image_agg", "metric", "source")
    }
    record.update(changes)
    record["scorer_params"].update(params)
    (tmp / f"{name}.json").write_text(json.dumps(record))


def test_records_are_flattened_with_metrics_in_percent(tmp_path: Path) -> None:
    write_record(tmp_path, "a")

    row = load_records(tmp_path).iloc[0]

    assert (row["family"], row["k"], row["layers"]) == ("few_abn", 5, "last_raw")
    assert row["image_auroc"] == pytest.approx(99.0)
    assert row["tau_val"] == pytest.approx(0.2)  # a score threshold, not a rate
    assert row["git_sha"] == "abcdef0"


def test_threshold_metrics_come_from_the_stored_scores_not_the_stored_values(
    tmp_path: Path,
) -> None:
    scores = {
        "val": {"score": [0.2, 0.4]},
        "test": {"score": [0.3, 0.5, 0.6, 0.1], "label": [0, 0, 1, 1]},
    }
    stale = {"image_auroc": 0.99, "fpr_at_tau_val": 0.0, "recall_at_tau_val": 1.0}
    write_record(tmp_path, "a", image_scores=scores, metrics=stale)

    row = load_records(tmp_path).iloc[0]

    assert row["tau_val"] == pytest.approx(0.4)
    assert row["fpr_at_tau_val"] == pytest.approx(50.0)
    assert row["recall_at_tau_val"] == pytest.approx(50.0)
    assert row["image_auroc"] == pytest.approx(
        99.0
    )  # not a threshold metric: kept as stored


def test_a_regime_without_a_shot_count_keeps_its_name_as_family(tmp_path: Path) -> None:
    write_record(tmp_path, "a", regime="oneclass")

    row = load_records(tmp_path).iloc[0]

    assert row["family"] == "oneclass" and pd.isna(row["k"])


def test_study_runs_drop_records_made_with_other_settings(tmp_path: Path) -> None:
    write_record(tmp_path, "study")
    write_record(tmp_path, "old_image_score", image_agg="max")
    write_record(tmp_path, "other_layer", layers=["8_raw"])
    write_record(tmp_path, "cropped", resize=256)
    write_record(tmp_path, "other_scorer", scorer="anomalydino")
    write_record(tmp_path, "wrn_with_dinov2_settings", backbone="wrn50_in1k")
    record = json.loads((tmp_path / "study.json").read_text())
    del record["image_scores"]
    record["config_hash"] = "no_scores"
    (tmp_path / "no_scores.json").write_text(json.dumps(record))

    study = study_runs(load_records(tmp_path))

    assert list(study["config_hash"]) == ["study"]
    assert list(study["bb"]) == ["S"]


def test_study_runs_can_take_another_layer(tmp_path: Path) -> None:
    write_record(tmp_path, "main")
    write_record(tmp_path, "mid", layers=["8_raw"])

    study = study_runs(load_records(tmp_path), layers={"dinov2_vits14": "8_raw"})

    assert list(study["config_hash"]) == ["mid"]


def test_a_rerun_on_a_changed_split_replaces_the_older_record(tmp_path: Path) -> None:
    write_record(
        tmp_path, "old", split_sha1="s0", timestamp="2026-10-01T00:00:00+00:00"
    )
    write_record(
        tmp_path, "new", split_sha1="s1", timestamp="2026-11-01T00:00:00+00:00"
    )
    write_record(tmp_path, "other_seed", seed=1, timestamp="2026-09-01T00:00:00+00:00")

    study = study_runs(load_records(tmp_path))

    assert sorted(study["config_hash"]) == ["new", "other_seed"]


def _grid(categories=CATS, seeds=(0, 1)) -> pd.DataFrame:
    rows = [
        {"bb": "S", "category": c, "seed": s, "x": 10.0 * i + s}
        for i, c in enumerate(categories)
        for s in seeds
    ]
    return pd.DataFrame(rows)


def test_complete_cells_pass() -> None:
    require_complete(_grid(), ["bb"], n_categories=3, seeds=(0, 1))


def test_a_missing_run_a_doubled_run_and_a_missing_category_are_refused() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="wrong seeds in \\['bottle'\\]"):
        require_complete(grid.iloc[1:], ["bb"], n_categories=3, seeds=(0, 1))
    with pytest.raises(ValueError, match="wrong seeds in \\['bottle'\\]"):
        require_complete(
            pd.concat([grid, grid.iloc[:1]]), ["bb"], n_categories=3, seeds=(0, 1)
        )
    with pytest.raises(ValueError, match="2 categories"):
        require_complete(
            grid[grid["category"] != "screw"], ["bb"], n_categories=3, seeds=(0, 1)
        )
    with pytest.raises(ValueError, match="no records"):
        require_complete(grid.iloc[:0], ["bb"])


def test_seed_summary_is_the_spread_over_seeds_of_the_category_mean() -> None:
    # category means per seed: seed 0 -> (0 + 10 + 20) / 3 = 10, seed 1 -> 11
    summary = seed_summary(_grid(), ["bb"], ["x"]).iloc[0]

    assert summary["x_mean"] == pytest.approx(10.5)
    assert summary["x_std"] == pytest.approx(
        0.5**0.5
    )  # sample standard deviation of 10, 11
    assert (summary["n_seeds"], summary["n_categories"], summary["n_runs"]) == (2, 3, 6)
