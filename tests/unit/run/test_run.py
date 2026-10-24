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


def _make_data_root(tmp_path: Path, n_good: int = 3, n_bad: int = 3) -> Path:
    """cat_a: 6 flat-grey train images; n_good grey + n_bad red-blob test images (with masks)."""
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
    for i in range(n_good + n_bad):
        add("test", i, i >= n_good)
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


@pytest.mark.filterwarnings("ignore:roi=.*fully masked")
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
    # the table is MVTec AD's: a same-named category of another dataset is not masked
    assert make_roi("pca_official_mvtec", "screw", patches, (2, 2), "mvtec_ad2") is None


def test_the_official_mask_is_refused_on_tokens_it_cannot_threshold(
    tmp_path: Path,
) -> None:
    from cmad.run import check_roi_layer

    check_roi_layer(None, "wrn50_in1k", "cnn_l23")  # no ROI: nothing to check
    check_roi_layer("pca_official", "dinov2_vits14", "last_raw")
    for backbone, layer in [
        ("dinov2_vits14", "last"),
        ("wrn50_in1k", "cnn_l23_pc_raw"),
    ]:
        with pytest.raises(ValueError, match="raw DINOv2 tokens"):
            check_roi_layer("pca_official_mvtec", backbone, layer)
    # and the runner enforces it before doing any work
    root = _make_data_root(tmp_path)
    cfg = _cfg(roi="pca_official", backbone={"name": "dinov2_fake"})  # layer "last"
    with pytest.raises(ValueError, match="raw DINOv2 tokens"):
        run_one(cfg, root, extractor_factory=lambda d: _Extractor())


def test_a_mask_that_empties_most_images_warns() -> None:
    from cmad.run import make_roi

    tokens = torch.ones(
        10, 4, 8
    )  # unit-scale tokens: nothing passes the absolute threshold
    with pytest.warns(UserWarning, match="fully masked"):
        make_roi("pca_official", "screw", tokens, (2, 2))


@pytest.mark.filterwarnings("ignore:roi=.*fully masked")
def test_rotated_references_and_roi_enter_the_run(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)
    plain, _ = run_one(_cfg(), root, extractor_factory=lambda d: _Extractor())
    cfg = _cfg(
        reference_rotations=[0, 90],
        roi="pca_official",
        interpolation="bicubic",
        backbone={"name": "dinov2_fake"},
        layers={"names": ["last_raw"]},
    )

    rec, _ = run_one(cfg, root, extractor_factory=lambda d: _Extractor())

    assert rec["bank"]["n_train_patches"] == 6 * 2 * 4  # images x angles x 2x2 grid
    assert rec["n_train"] == 6
    assert rec["reference_rotations"] == [0.0, 90.0] and rec["roi"] == "pca_official"
    assert rec["interpolation"] == "bicubic"
    assert rec["config_hash"] != plain["config_hash"]
    assert "reference_rotations" not in plain and "roi" not in plain  # defaults stay out of the identity


def _neg_cfg(**scorer_over) -> dict:
    scorer = {
        "name": "negbank",
        "k": 1,
        "coreset_fraction": 1.0,
        "coreset_proj_dim": 8,
        "image_agg": "max",
        "metric": "euclidean",
        "source": "gt_mask",
        "top_frac": 0.25,
        "lam": 0.0,
        "lambdas": [0.0, 0.5, 1.0, 2.0],
        **scorer_over,
    }
    return _cfg(scorer=scorer, **_regime("few_abn_k1"))


def test_negbank_needs_labelled_support_in_the_regime(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path, 4, 8)
    cfg = _neg_cfg()
    cfg["regime"] = {"name": "oneclass_full"}
    with pytest.raises(ValueError, match="few_abn_k"):
        run_one(cfg, root, extractor_factory=lambda d: _Extractor())


def test_negbank_run_records_bank_curve_oracle_and_diagnostic(tmp_path: Path) -> None:
    root = _make_data_root(
        tmp_path, 4, 8
    )  # 8 faults of one type clear the k + 5 guard at k = 1
    knn_cfg = _cfg(**_regime("few_abn_k1"))
    plain, _ = run_one(knn_cfg, root, extractor_factory=lambda d: _Extractor())

    rec, _ = run_one(_neg_cfg(), root, extractor_factory=lambda d: _Extractor())

    nb = rec["neg_bank"]
    assert (
        rec["split"]["n_support"] == 1 and rec["n_anomalous"] == 7
    )  # the support left the test set
    assert nb["active"] and nb["source"] == "gt_mask" and nb["lambda"] == 0.0
    assert nb["selected_lambda"] is None  # no selection rule yet
    assert list(nb["curve"]) == ["0", "0.5", "1", "2"]
    assert set(nb["curve"]["0"]) == {"image_auroc", "tpr_at_1pct_fpr"}
    assert nb["oracle"]["image_auroc"] == max(
        m["image_auroc"] for m in nb["curve"].values()
    )
    (entry,) = nb["support"].values()
    assert entry["defect_type"] == "blob" and entry["n_negative_patches"] > 0
    assert nb["n_negative_patches"] == entry["n_negative_patches"] > 0
    # lambda = 0 is plain kNN on the very same split
    assert rec["metrics"]["image_auroc"] == plain["metrics"]["image_auroc"]
    assert nb["curve"]["0"]["image_auroc"] == plain["metrics"]["image_auroc"]
    # the bank was built from the GT patches, so the diagnostic is exact
    assert (
        nb["selection"]["all"]["precision"] == 1.0
        and nb["selection"]["all"]["recall"] == 1.0
    )
    assert "neg_bank" not in plain
    assert nb["selection"]["all"]["n_defect_in_roi"] is None  # no ROI in this run
    assert nb["top_frac"] is None  # gt_mask does not read it


def test_negbank_sources_and_operating_lambda_enter_the_identity(
    tmp_path: Path,
) -> None:
    root = _make_data_root(tmp_path, 4, 8)
    records = {}
    for name, over in {
        "gt": {},
        "top_q": {"source": "top_q"},
        "random": {"source": "random"},
        "lam1": {"lam": 1.0},
    }.items():
        records[name], _ = run_one(
            _neg_cfg(**over), root, extractor_factory=lambda d: _Extractor()
        )

    assert len({r["config_hash"] for r in records.values()}) == 4
    assert records["lam1"]["neg_bank"]["lambda"] == 1.0
    top = records["top_q"]["neg_bank"]["selection"]["all"]
    assert top["n_selected"] == records["top_q"]["neg_bank"]["n_negative_patches"]
    assert 0.0 <= top["precision"] <= 1.0 and 0.0 <= top["recall"] <= 1.0
    # the red blob is the only thing far from the grey bank, so top_q finds it
    assert top["recall"] > 0.0


def test_negbank_with_no_usable_support_falls_back_loudly(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path)  # 3 faults: below k + 5, so no support is drawn

    with pytest.warns(UserWarning, match="plain kNN"):
        rec, _ = run_one(_neg_cfg(), root, extractor_factory=lambda d: _Extractor())

    nb = rec["neg_bank"]
    assert (
        nb["active"] is False and nb["n_negative_patches"] == 0 and nb["curve"] is None
    )
    assert rec["split"]["skipped_support_types"] == ["blob"]


@pytest.mark.filterwarnings("ignore:roi=.*fully masked")
def test_negbank_with_roi_runs_and_floors_the_patch_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cmad.run as run_module

    calls = []
    real = run_module.floor_masked
    monkeypatch.setattr(
        run_module,
        "floor_masked",
        lambda maps, roi: calls.append((maps.shape, roi.shape)) or real(maps, roi),
    )
    root = _make_data_root(tmp_path, 4, 8)
    cfg = {
        **_neg_cfg(lam=1.0),
        "roi": "pca_official",
        "backbone": {"name": "dinov2_fake"},
        "layers": {"names": ["last_raw"]},
    }

    rec, _ = run_one(cfg, root, extractor_factory=lambda d: _Extractor())

    assert rec["roi"] == "pca_official" and rec["neg_bank"]["active"]
    sel = rec["neg_bank"]["selection"]["all"]
    assert 0 <= sel["n_defect_in_roi"] <= sel["n_defect"]
    assert sel["recall_in_roi"] is None or 0.0 <= sel["recall_in_roi"] <= 1.0
    assert calls == [
        ((11, 2, 2), (11, 1, 2, 2))
    ]  # all test maps at once, before the pixel metrics
    assert np.isfinite(rec["metrics"]["pixel_auroc"])


def test_negbank_config_composes_with_a_support_regime() -> None:
    from hydra import compose, initialize_config_dir

    configs = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(config_dir=configs, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["category=x", "scorer=negbank", "regime.name=few_abn_k5"],
        )

    assert cfg.scorer.name == "negbank" and cfg.scorer.source == "gt_mask"
    assert next(iter(cfg.scorer.lambdas)) == 0.0 and cfg.scorer.lam == "select"
    assert cfg.regime.name == "few_abn_k5"


def test_top_frac_is_part_of_the_identity_only_for_sources_that_read_it(
    tmp_path: Path,
) -> None:
    root = _make_data_root(tmp_path, 4, 8)

    def run(**over):
        return run_one(_neg_cfg(**over), root, extractor_factory=lambda d: _Extractor())

    gt_a, _ = run(top_frac=0.25)
    gt_b, ran = run(top_frac=0.10)
    top_a, _ = run(source="top_q", top_frac=0.25)
    top_b, _ = run(source="top_q", top_frac=0.10)

    assert (
        gt_a["config_hash"] == gt_b["config_hash"] and not ran
    )  # the second one is cached
    assert top_a["config_hash"] != top_b["config_hash"]
    assert "top_frac" not in gt_a["scorer_params"]
    assert top_a["neg_bank"]["top_frac"] == 0.25


def test_gather_returns_features_in_id_order_across_original_splits(
    tmp_path: Path,
) -> None:
    from cmad.acquire import PreprocessConfig
    from cmad.run import _gather, _shard_dirs

    root = _make_data_root(tmp_path, 4, 8)
    run_one(_cfg(), root, extractor_factory=lambda d: _Extractor())  # fills the shards
    manifest = pd.read_parquet(root / "manifests" / "ds.parquet")
    origin = manifest.set_index("image_id")["orig_split"]
    pre = PreprocessConfig(res=16, resize=None, interpolation="bilinear")
    layers = ["last"]
    dirs = {
        o: _shard_dirs(root, manifest, "cat_a", o, "fake", layers, pre)
        for o in ("train", "test")
    }

    ids = [
        "ds/cat_a/test/005.png",
        "ds/cat_a/train/002.png",
        "ds/cat_a/test/001.png",
        "ds/cat_a/train/000.png",
        "ds/cat_a/test/009.png",
    ]
    together = _gather(dirs, origin, ids, layers)

    for position, image_id in enumerate(
        ids
    ):  # one image at a time has no ordering to get wrong
        alone = _gather(dirs, origin, [image_id], layers)
        assert torch.equal(together.patches["last"][position], alone.patches["last"][0])
        assert torch.equal(together.global_token[position], alone.global_token[0])
    # and the images really differ, so the comparison above can detect a mix-up
    assert not torch.equal(together.patches["last"][0], together.patches["last"][1])


def _select_cfg(**over) -> dict:
    cfg = _neg_cfg(lam="select", **over)
    cfg["regime"] = {"name": "few_abn_k2"}
    return cfg


def test_negbank_selects_lambda_with_leave_one_support_out(tmp_path: Path) -> None:
    root = _make_data_root(
        tmp_path, 4, 12
    )  # 12 faults: the k + 5 guard passes at k = 2

    with pytest.warns(UserWarning, match="only 1 validation images"):
        rec, _ = run_one(_select_cfg(), root, extractor_factory=lambda d: _Extractor())

    nb = rec["neg_bank"]
    rule = nb["selection_rule"]
    assert rec["split"]["n_support"] == 2 and rule["selected"] and rule["n_folds"] == 2
    assert rule["n_val"] == rec["split"]["n_val"] == 1  # 10 % of the 6 training images
    assert "vs_lambda0" in rule
    assert list(rule["curve"]) == ["0", "0.5", "1", "2"]
    assert nb["selected_lambda"] == nb["lambda"] == rule["lambda"]
    # the operating lambda is the one scored on the test images, so the record agrees with itself
    assert nb["selected_image_auroc"] == nb["curve"][f"{nb['lambda']:g}"]["image_auroc"]
    assert rec["metrics"]["image_auroc"] == nb["selected_image_auroc"]
    assert nb["oracle_gap"] == pytest.approx(
        nb["oracle"]["image_auroc"] - nb["selected_image_auroc"]
    )
    assert nb["oracle_gap"] >= 0.0


def test_a_fixed_lambda_records_no_selection(tmp_path: Path) -> None:
    root = _make_data_root(tmp_path, 4, 12)
    cfg = _select_cfg()
    cfg["scorer"] = {**cfg["scorer"], "lam": 1.0}

    rec, _ = run_one(cfg, root, extractor_factory=lambda d: _Extractor())

    nb = rec["neg_bank"]
    assert (
        nb["lambda"] == 1.0
        and nb["selected_lambda"] is None
        and nb["selection_rule"] is None
    )
    assert "oracle_gap" not in nb


def test_selection_with_a_single_support_image_falls_back_to_plain_knn(
    tmp_path: Path,
) -> None:
    root = _make_data_root(tmp_path, 4, 8)  # few_abn_k1: one support image

    rec, _ = run_one(
        {**_neg_cfg(lam="select")}, root, extractor_factory=lambda d: _Extractor()
    )

    nb = rec["neg_bank"]
    assert nb["selection_rule"]["selected"] is False
    assert "fewer than two" in nb["selection_rule"]["reason"]
    assert nb["lambda"] == 0.0 and nb["selected_lambda"] is None
