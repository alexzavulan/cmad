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
                "dataset": "ds",
                "category": "cat_a",
                "condition": None,
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
        "regime": {"name": "oneclass_full"},
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


def test_metric_auto_follows_the_layer_and_cosine_on_raw_is_refused() -> None:
    from cmad.run import resolve_metric

    assert resolve_metric("auto", ["cnn_l23_pc_raw"]) == "euclidean"
    assert resolve_metric("auto", ["last"]) == "cosine"
    assert resolve_metric("sqeuclidean", ["last"]) == "sqeuclidean"
    with pytest.raises(ValueError, match="raw"):
        resolve_metric("cosine", ["cnn_l23_raw"])
    with pytest.raises(ValueError, match="mixed"):
        resolve_metric("auto", ["cnn_l23_raw", "cnn_l23"])


def test_backbone_recipes_compose_from_the_shipped_configs() -> None:
    from hydra import compose, initialize_config_dir

    configs = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(config_dir=configs, version_base=None):
        wrn = compose(
            config_name="config", overrides=["category=x", "backbone=wrn50_in1k"]
        )
        wrn_cos = compose(
            config_name="config",
            overrides=["category=x", "backbone=wrn50_in1k", "layers=cnn_l23"],
        )
        dino = compose(config_name="config", overrides=["category=x"])

    assert list(wrn.layers.names) == ["cnn_l23_pc_raw"]  # the recipe's default
    assert list(wrn_cos.layers.names) == ["cnn_l23"]  # the command line still wins
    assert list(dino.layers.names) == ["last"] and dino.backbone.name == "dinov2_vits14"
    assert wrn.scorer.metric == "auto"


def _regime(name: str, **extra) -> dict:
    return {"regime": {"name": name, **extra}}


def test_oneclass_holds_out_validation_and_reports_threshold_metrics(
    tmp_path: Path,
) -> None:
    root = _make_data_root(tmp_path)

    record, _ = run_one(
        _cfg(**_regime("oneclass")), root, extractor_factory=lambda d: _Extractor()
    )

    assert (
        record["split"]["n_val"] == 1 and record["n_train"] == 5
    )  # 10 % of 6, rounded
    assert record["bank"]["n_patches"] == 5 * 4
    m = record["metrics"]
    assert m["tau_val"] is not None
    for key in (
        "f1_at_tau_val",
        "precision_at_tau_val",
        "recall_at_tau_val",
        "fpr_at_tau_val",
    ):
        assert 0.0 <= m[key] <= 1.0
    saved = root / "splits" / "ds" / "cat_a" / "oneclass_seed0.json"
    assert saved.exists() and json.loads(saved.read_text())["val_normal"]


def test_oneclass_full_has_no_validation_set_and_null_threshold_metrics(
    tmp_path: Path,
) -> None:
    root = _make_data_root(tmp_path)

    record, _ = run_one(_cfg(), root, extractor_factory=lambda d: _Extractor())

    assert record["split"]["n_val"] == 0 and record["n_train"] == 6
    assert record["metrics"]["tau_val"] is None
    assert record["metrics"]["f1_at_tau_val"] is None


def test_few_norm_contiguous_blocks_fit_on_the_official_slice(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    cfg = _cfg(**_regime("few_norm_k2", shot_selection="contiguous_blocks"), seed=1)

    record, _ = run_one(cfg, root, extractor_factory=lambda d: _Extractor())

    assert record["n_train"] == 2 and record["bank"]["n_patches"] == 2 * 4
    split = json.loads(
        (
            root
            / "splits"
            / "ds"
            / "cat_a"
            / "few_norm_k2+contiguous_blocks_seed1.json"
        ).read_text()
    )
    assert split["train_normal"] == [f"ds/cat_a/train/{i:03d}.png" for i in (2, 3)]


def test_few_abn_guard_is_recorded_in_the_run(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)  # 3 anomalies of one type: below k + 5

    record, _ = run_one(
        _cfg(**_regime("few_abn_k1")), root, extractor_factory=lambda d: _Extractor()
    )

    assert record["split"]["skipped_support_types"] == ["blob"]
    assert record["split"]["n_support"] == 0 and record["n_anomalous"] == 3


def test_the_regime_is_part_of_the_run_identity(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    full, _ = run_one(_cfg(), root, extractor_factory=lambda d: _Extractor())
    held_out, _ = run_one(_cfg(**_regime("oneclass")), root)

    assert full["config_hash"] != held_out["config_hash"]
    assert full["split_sha1"] != held_out["split_sha1"]


def test_resolve_metric_with_a_normalising_scorer_and_make_roi_modes() -> None:
    from cmad.run import make_roi, resolve_metric

    assert resolve_metric("auto", ["last_raw"], normalize=True) == "cosine"
    assert resolve_metric("cosine", ["last_raw"], normalize=True) == "cosine"
    with pytest.raises(ValueError, match="raw"):
        resolve_metric("cosine", ["last_raw"], normalize=False)
    patches = torch.zeros(1, 4, 3)
    assert make_roi(None, "screw", patches, (2, 2)) is None
    assert make_roi("pca_official_mvtec", "bottle", patches, (2, 2)) is None  # not in its table
    assert make_roi("pca_official_mvtec", "screw", patches, (2, 2)) is not None
    with pytest.raises(ValueError, match="unknown roi"):
        make_roi("magic", "screw", patches, (2, 2))


def test_rotated_references_and_roi_enter_the_run(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    plain, _ = run_one(_cfg(), root, extractor_factory=lambda d: _Extractor())
    cfg = _cfg(reference_rotations=[0, 90], roi="pca_official", interpolation="bicubic")

    rec, _ = run_one(cfg, root, extractor_factory=lambda d: _Extractor())

    assert rec["bank"]["n_train_patches"] == 6 * 2 * 4  # images x angles x 2x2 grid
    assert rec["n_train"] == 6
    assert rec["reference_rotations"] == [0.0, 90.0] and rec["roi"] == "pca_official"
    assert rec["interpolation"] == "bicubic"
    assert rec["config_hash"] != plain["config_hash"]
    assert "reference_rotations" not in plain and "roi" not in plain  # defaults stay out of the identity
