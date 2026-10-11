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
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from cmad.acquire import PreprocessConfig
from cmad.eval import evaluate_category
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

SPLITS = ("train", "test")


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


def _factors(cfg: Mapping, res: int, with_aupro: bool) -> dict:
    """Everything that determines a run's numbers (and nothing about where it ran)."""
    scorer = {k: v for k, v in cfg["scorer"].items() if k != "name"}
    scorer["metric"] = resolve_metric(scorer["metric"], list(cfg["layers"]["names"]))
    return {
        "dataset": cfg["dataset"],
        "category": cfg["category"],
        "regime": cfg["regime"]["name"],
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


def _load(directories: dict[str, Path]) -> FeatureBatch:
    shards = {layer: Shard(d) for layer, d in directories.items()}
    first = next(iter(shards.values()))
    return FeatureBatch(
        patches={layer: s.patches() for layer, s in shards.items()},
        grid=first.grid,
        global_token=first.global_token(),
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
    factors = _factors(cfg, res, with_aupro)
    digest = config_hash(factors)
    results_dir = data_root / "results" / "runs"

    path = run_path(results_dir, digest)
    if path.exists() and not cfg["force"]:
        return json.loads(path.read_text()), False

    start = time.perf_counter()
    seed_everything(factors["seed"])
    device = auto_device() if cfg["device"] == "auto" else cfg["device"]
    manifest = pd.read_parquet(data_root / "manifests" / f"{cfg['dataset']}.parquet")
    rows = {s: category_rows(manifest, factors["category"], s) for s in SPLITS}
    preprocess = PreprocessConfig(res=res, resize=resize)
    layers = factors["layers"]
    directories = {
        s: shard_directories(
            features_dir=data_root / "features",
            backbone=backbone,
            layers=layers,
            preprocess=preprocess,
            category=factors["category"],
            split=s,
            image_ids=tuple(rows[s]["image_id"]),
        )
        for s in SPLITS
    }

    # compute on miss: the extractor (and its weights) is only built if a shard is absent
    missing = [s for s in SPLITS if not all(map(Shard.exists, directories[s].values()))]
    extract_s = 0.0
    if missing:
        t = time.perf_counter()
        make = extractor_factory or (lambda d: build_extractor(backbone, device=d))
        extractor = make(device)
        for s in missing:
            extract_to_cache(
                manifest=manifest,
                raw_root=data_root / "raw",
                features_dir=data_root / "features",
                extractor=extractor,
                backbone=backbone,
                layers=layers,
                preprocess=preprocess,
                category=factors["category"],
                split=s,
            )
        extract_s = time.perf_counter() - t

    train, test = _load(directories["train"]), _load(directories["test"])
    n_train = len(rows["train"])
    n_train_patches = n_train * train.grid[0] * train.grid[1]

    sc = factors["scorer_params"]
    scorer = KNNScorer(
        k=sc["k"],
        coreset_fraction=sc["coreset_fraction"],
        coreset_proj_dim=sc["coreset_proj_dim"],
        image_agg=sc["image_agg"],
        metric=sc["metric"],
        device=device,
        seed=factors["seed"],
    )
    t = time.perf_counter()
    scorer.fit(train)
    fit_s = time.perf_counter() - t

    t = time.perf_counter()
    out = scorer.score(test)
    score_s = time.perf_counter() - t

    t = time.perf_counter()
    metrics = evaluate_category(
        out,
        rows["test"],
        data_root / "raw",
        res=res,
        resize=resize,
        sigma=factors["sigma"],
        with_aupro=with_aupro,
        fpr_limit=factors["fpr_limit"],
    )
    metrics.setdefault(f"aupro_{factors['fpr_limit']:g}", None)
    eval_s = time.perf_counter() - t

    record = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": digest[:12],
        "config_hash": digest,
        **provenance(),
        "device": device,
        **factors,
        "n_train": n_train,
        "n_test": metrics.pop("n_test"),
        "n_anomalous": metrics.pop("n_anomalous"),
        "grid": list(train.grid),
        "cache": {
            s: {layer: d.name for layer, d in directories[s].items()} for s in SPLITS
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
