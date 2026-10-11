import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from cmad.features import FeatureBatch, FeatureExtractor
from cmad.results import config_hash
from cmad.run import run_one


class _Extractor(FeatureExtractor):
    """2x2 average-pooled pixels as 'patches'; counts extract() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def extract(self, x: torch.Tensor, layers: list[str]) -> FeatureBatch:
        self.calls += 1
        pooled = F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)
        return FeatureBatch(
            patches={layer: pooled.half() for layer in layers},
            grid=(2, 2),
            global_token=x.mean(dim=(2, 3)).half(),
        )


def _make_data_root(tmp_path: Path) -> Path:
    """cat_a: 6 flat-grey train images; 3 grey + 3 red-blob test images (with masks)."""
    root = tmp_path / "data"
    rng = np.random.default_rng(0)
    rows = []

    def add(split: str, i: int, anomalous: bool) -> None:
        stem = f"ds/cat_a/{split}/{i:03d}"
        arr = np.full((20, 30, 3), 110, np.uint8)
        arr += rng.integers(0, 10, arr.shape, dtype=np.uint8)
        mask_path = None
        if anomalous:
            arr[4:16, 9:21] = (230, 20, 20)
            mask = np.zeros((20, 30), np.uint8)
            mask[4:16, 9:21] = 255
            mask_path = f"{stem}_mask.png"
            Image.fromarray(mask).save(root / "raw" / mask_path)
        Image.fromarray(arr).save(root / "raw" / f"{stem}.png")
        rows.append(
            {
                "image_id": f"{stem}.png",
                "image_path": f"{stem}.png",
                "mask_path": mask_path,
                "category": "cat_a",
                "orig_split": split,
                "label": int(anomalous),
                "defect_type": "blob" if anomalous else "good",
            }
        )

    (root / "raw" / "ds" / "cat_a" / "train").mkdir(parents=True)
    (root / "raw" / "ds" / "cat_a" / "test").mkdir(parents=True)
    for i in range(6):
        add("train", i, False)
    for i in range(6):
        add("test", i, i >= 3)
    (root / "manifests").mkdir()
    pd.DataFrame(rows).to_parquet(root / "manifests" / "ds.parquet")
    return root


def _cfg(**over) -> dict:
    cfg = {
        "dataset": "ds",
        "category": "cat_a",
        "seed": 0,
        "res": 16,
        "resize": None,
        "sigma": 1.0,
        "with_aupro": False,
        "fpr_limit": 0.3,
        "device": "cpu",
        "force": False,
        "backbone": {"name": "fake"},
        "layers": {"names": ["last"]},
        "regime": {"name": "oneclass"},
        "scorer": {
            "name": "knn",
            "k": 1,
            "coreset_fraction": 1.0,
            "coreset_proj_dim": 8,
            "image_agg": "max",
            "metric": "euclidean",
        },
    }
    cfg.update(over)
    return cfg


def test_run_writes_a_complete_record_and_is_idempotent(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    ex = _Extractor()

    record, ran = run_one(_cfg(), root, extractor_factory=lambda device: ex)

    assert ran and ex.calls > 0
    path = root / "results" / "runs" / f"{record['config_hash']}.json"
    assert json.loads(path.read_text()) == record
    assert list(path.parent.glob("*.tmp")) == []
    assert record["metrics"]["image_auroc"] == 1.0
    assert record["metrics"]["aupro_0.3"] is None  # not evaluated, not zero
    assert (record["n_train"], record["n_test"], record["n_anomalous"]) == (6, 6, 3)
    assert record["bank"]["n_patches"] == record["bank"]["n_train_patches"] == 6 * 4
    assert record["scorer_params"]["metric"] == "euclidean"
    for key in ("git_sha", "hostname", "timestamp", "cache", "timing"):
        assert key in record

    again, ran_again = run_one(_cfg(), root, extractor_factory=lambda device: ex)

    assert not ran_again and again == record


def test_cache_hit_never_builds_the_extractor(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    run_one(_cfg(), root, extractor_factory=lambda device: _Extractor())

    def explode(device: str) -> FeatureExtractor:
        raise AssertionError("extractor built despite a complete cache")

    # a different scorer (new hash) over the same features
    cfg = _cfg(scorer={**_cfg()["scorer"], "k": 3})
    record, ran = run_one(cfg, root, extractor_factory=explode)

    assert ran and record["scorer_params"]["k"] == 3
    assert record["timing"]["extract_s"] == 0.0


def test_force_recomputes(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    run_one(_cfg(), root, extractor_factory=lambda device: _Extractor())

    _, ran = run_one(_cfg(force=True), root)

    assert ran


def test_config_hash_tracks_factors_only(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    base, _ = run_one(_cfg(), root, extractor_factory=lambda device: _Extractor())
    other_seed, _ = run_one(_cfg(seed=1), root)

    assert other_seed["config_hash"] != base["config_hash"]
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    # device and force are not identity: same hash, found as cached
    same, ran = run_one(_cfg(device="cpu", force=False), root)
    assert not ran and same["config_hash"] == base["config_hash"]


def test_unknown_category_fails_loudly(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    with pytest.raises(ValueError, match="no manifest rows"):
        run_one(_cfg(category="nope"), root)
