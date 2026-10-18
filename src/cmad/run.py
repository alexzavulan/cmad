"""One run = one (category, config): cached features -> fit -> score -> metrics -> one JSON file.

    python -m cmad.run category=bottle backbone=wrn50_in1k layers=cnn_l23_raw \
        scorer=patchcore res=224 resize=256
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from cmad.acquire import PreprocessConfig
from cmad.data.regimes import Split, get_split, parse_regime, split_path
from cmad.eval import evaluate_category, threshold_metrics
from cmad.features import (
    FeatureBatch,
    FeatureExtractor,
    Shard,
    build_extractor,
    category_rows,
    default_resolution,
    extract_to_cache,
    shard_directories,
)
from cmad.results import (
    RESULT_SCHEMA_VERSION,
    config_hash,
    provenance,
    run_path,
    write_run,
)
from cmad.scorers import KNNScorer
from cmad.utils import auto_device, seed_everything

TAU_QUANTILE = 0.99  # tau_val: the 99th percentile of validation-normal image scores


def _resolve_aupro(setting: object) -> bool:
    if setting == "auto":
        available = importlib.util.find_spec("anomalib") is not None
        if not available:
            warnings.warn("anomalib not installed: AUPRO is recorded as null")
        return available
    return bool(setting)


def resolve_metric(metric: str, layers: list[str]) -> str:
    """ "auto" follows the layer: raw (un-normalised) layers keep their activation magnitudes and
    are compared with Euclidean distance, normalised layers with cosine. Cosine assumes unit
    vectors, so asking for it on a raw layer is an error rather than a silent wrong number."""
    raw = [layer.endswith("_raw") for layer in layers]
    if metric == "auto":
        if any(raw) and not all(raw):
            raise ValueError(f"cannot pick a metric for mixed raw/normalised {layers}")
        return "euclidean" if all(raw) else "cosine"
    if metric == "cosine" and any(raw):
        raise ValueError(f"cosine needs unit vectors but {layers} contains raw layers")
    return metric


def _factors(cfg: Mapping, res: int, with_aupro: bool, split: Split) -> dict:
    """Everything that determines a run's numbers (and nothing about where it ran)."""
    scorer = {k: v for k, v in cfg["scorer"].items() if k != "name"}
    scorer["metric"] = resolve_metric(scorer["metric"], list(cfg["layers"]["names"]))
    return {
        "dataset": cfg["dataset"],
        "category": cfg["category"],
        "regime": split.spec.name,
        "regime_spec": asdict(split.spec),
        "split_sha1": split.sha1(),
        "seed": int(cfg["seed"]),
        "backbone": cfg["backbone"]["name"],
        "layers": list(cfg["layers"]["names"]),
        "res": res,
        "resize": cfg["resize"],
        "scorer": cfg["scorer"]["name"],
        "scorer_params": scorer,
        "sigma": float(cfg["sigma"]),
        "with_aupro": with_aupro,
        "fpr_limit": float(cfg["fpr_limit"]),
        "corruption": "none",
    }


def _shard_dirs(
    data_root: Path,
    manifest: pd.DataFrame,
    category: str,
    orig_split: str,
    backbone: str,
    layers: list[str],
    preprocess: PreprocessConfig,
) -> dict[str, Path]:
    rows = category_rows(manifest, category, orig_split)
    return shard_directories(
        features_dir=data_root / "features",
        backbone=backbone,
        layers=layers,
        preprocess=preprocess,
        category=category,
        split=orig_split,
        image_ids=tuple(rows["image_id"]),
    )


def _gather(
    dirs: dict[str, dict[str, Path]],
    origin: pd.Series,
    ids: list[str],
    layers: list[str],
) -> FeatureBatch:
    """Features of exactly `ids`, in that order, read from the per-original-split shards."""
    positions: dict[str, list[int]] = {}
    for pos, image_id in enumerate(ids):
        positions.setdefault(origin[image_id], []).append(pos)
    order = [p for group in positions.values() for p in group]
    inverse = torch.as_tensor(np.argsort(order))
    parts: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
    parts_global: list[torch.Tensor] = []
    grid = None
    for orig_split, group in positions.items():
        wanted = [ids[p] for p in group]
        shards = {layer: Shard(dirs[orig_split][layer]) for layer in layers}
        grid = grid or shards[layers[0]].grid
        for layer in layers:
            parts[layer].append(shards[layer].patches(wanted))
        parts_global.append(shards[layers[0]].global_token(wanted))
    assert grid is not None
    return FeatureBatch(
        patches={layer: torch.cat(chunks)[inverse] for layer, chunks in parts.items()},
        grid=grid,
        global_token=torch.cat(parts_global)[inverse],
    )


def run_one(
    cfg: Mapping,
    data_root: Path,
    extractor_factory: Callable[[str], FeatureExtractor] | None = None,
) -> tuple[dict, bool]:
    """Run (or find the finished result of) one config. Returns (record, ran)."""
    data_root = Path(data_root)
    backbone = cfg["backbone"]["name"]
    res = int(cfg["res"] or default_resolution(backbone))
    resize = cfg["resize"]
    with_aupro = _resolve_aupro(cfg["with_aupro"])
    category, seed = cfg["category"], int(cfg["seed"])

    manifest = pd.read_parquet(data_root / "manifests" / f"{cfg['dataset']}.parquet")
    reg = cfg["regime"]
    spec = parse_regime(
        reg["name"],
        shot_selection=reg.get("shot_selection"),
        val_normal_frac=reg.get("val_normal_frac"),
        train_condition=reg.get("train_condition"),
    )
    splits_dir = data_root / "splits"
    split = get_split(manifest, category, spec, seed, splits_dir)

    factors = _factors(cfg, res, with_aupro, split)
    digest = config_hash(factors)
    results_dir = data_root / "results" / "runs"

    path = run_path(results_dir, digest)
    if path.exists() and not cfg["force"]:
        return json.loads(path.read_text()), False

    start = time.perf_counter()
    seed_everything(seed)
    device = auto_device() if cfg["device"] == "auto" else cfg["device"]
    preprocess = PreprocessConfig(res=res, resize=resize)
    layers = factors["layers"]

    support_ids = [i for ids in split.support_abnormal.values() for i in ids]
    origin = manifest.set_index("image_id")["orig_split"]
    needed = sorted(
        set(origin[split.train_normal + split.val_normal + support_ids + split.test])
    )
    directories = {
        o: _shard_dirs(data_root, manifest, category, o, backbone, layers, preprocess)
        for o in needed
    }

    # compute on miss: the extractor (and its weights) is only built if a shard is absent
    missing = [o for o in needed if not all(map(Shard.exists, directories[o].values()))]
    extract_s = 0.0
    if missing:
        t = time.perf_counter()
        make = extractor_factory or (lambda d: build_extractor(backbone, device=d))
        extractor = make(device)
        for o in missing:
            extract_to_cache(
                manifest=manifest,
                raw_root=data_root / "raw",
                features_dir=data_root / "features",
                extractor=extractor,
                backbone=backbone,
                layers=layers,
                preprocess=preprocess,
                category=category,
                split=o,
            )
        extract_s = time.perf_counter() - t

    train = _gather(directories, origin, split.train_normal, layers)
    test = _gather(directories, origin, split.test, layers)
    n_train_patches = len(split.train_normal) * train.grid[0] * train.grid[1]

    sc = factors["scorer_params"]
    scorer = KNNScorer(
        k=sc["k"],
        coreset_fraction=sc["coreset_fraction"],
        coreset_proj_dim=sc["coreset_proj_dim"],
        image_agg=sc["image_agg"],
        metric=sc["metric"],
        device=device,
        seed=seed,
    )
    t = time.perf_counter()
    scorer.fit(train)  # support abnormals are not used until the negative bank exists
    fit_s = time.perf_counter() - t

    t = time.perf_counter()
    out = scorer.score(test)
    tau = None
    if split.val_normal:
        val = _gather(directories, origin, split.val_normal, layers)
        tau = float(np.quantile(scorer.score(val).image_score.numpy(), TAU_QUANTILE))
    score_s = time.perf_counter() - t

    t = time.perf_counter()
    test_rows = manifest.set_index("image_id").loc[split.test].reset_index()
    metrics = evaluate_category(
        out,
        test_rows,
        data_root / "raw",
        res=res,
        resize=resize,
        sigma=factors["sigma"],
        with_aupro=with_aupro,
        fpr_limit=factors["fpr_limit"],
    )
    metrics.setdefault(f"aupro_{factors['fpr_limit']:g}", None)
    deployable = (
        "f1_at_tau_val",
        "precision_at_tau_val",
        "recall_at_tau_val",
        "fpr_at_tau_val",
    )
    metrics["tau_val"] = tau
    metrics.update(dict.fromkeys(deployable))  # null when there is no validation set
    if tau is not None:
        metrics.update(
            threshold_metrics(
                out.image_score.numpy(), test_rows["label"].to_numpy(), tau
            )
        )
    eval_s = time.perf_counter() - t

    record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": digest[:12],
        "config_hash": digest,
        **provenance(),
        "device": device,
        **factors,
        "n_train": len(split.train_normal),
        "n_test": metrics.pop("n_test"),
        "n_anomalous": metrics.pop("n_anomalous"),
        "split": {
            "sha1": split.sha1(),
            "file": split_path(Path("splits"), split).as_posix(),
            "n_train": len(split.train_normal),
            "n_val": len(split.val_normal),
            "n_support": len(support_ids),
            "n_test": len(split.test),
            "skipped_support_types": split.skipped_support_types,
        },
        "grid": list(train.grid),
        "cache": {
            o: {layer: d.name for layer, d in directories[o].items()} for o in needed
        },
        "bank": {
            "n_train_patches": n_train_patches,
            "n_patches": int(scorer.bank.shape[0]),
            "dim": int(scorer.bank.shape[1]),
            "dtype": str(scorer.bank.dtype),
            "bytes": scorer.memory_bytes(),
        },
        "metrics": metrics,
        "timing": {
            "extract_s": extract_s,
            "fit_s": fit_s,
            "score_s": score_s,
            "eval_s": eval_s,
            "wall_s": time.perf_counter() - start,
        },
    }
    write_run(results_dir, record)
    return record, True


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    root = cfg.data_root or os.environ.get("CMAD_DATA_ROOT")
    if not root:
        raise SystemExit("CMAD_DATA_ROOT is not set; export it or pass data_root=...")
    plain = OmegaConf.to_container(cfg, resolve=True)
    record, ran = run_one(plain, Path(root).expanduser())  # type: ignore[arg-type]
    m = record["metrics"]
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    print(
        f"{'ran' if ran else 'cached'} {record['run_id']} "
        f"{record['category']} {record['backbone']}/{'+'.join(record['layers'])} "
        f"{record['scorer']}: image_auroc={fmt(m['image_auroc'])} "
        f"pixel_auroc={fmt(m['pixel_auroc'])} "
        f"aupro={fmt(m.get(f'aupro_{record["fpr_limit"]:g}'))} "
        f"({record['timing']['wall_s']:.1f}s)"
    )


if __name__ == "__main__":
    main()
