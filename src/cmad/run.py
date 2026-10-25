"""One run = one (category, config): cached features -> fit -> score -> metrics -> one JSON file.

    python -m cmad.run category=bottle backbone=wrn50_in1k layers=cnn_l23_raw \
        scorer=patchcore res=224 resize=256
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from cmad.acquire import (
    MVTEC_MASKED_CATEGORIES,
    ROTATIONS_OFFICIAL,
    OfficialRoiMask,
    PreprocessConfig,
)
from cmad.data.regimes import Split, get_split, parse_regime, split_path
from cmad.eval import (
    coverage_to_patches,
    evaluate_category,
    image_metrics,
    mask_coverage,
    threshold_metrics,
)
from cmad.features import (
    FeatureBatch,
    FeatureExtractor,
    Shard,
    build_extractor,
    category_rows,
    default_resolution,
    extract_rotated,
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
from cmad.scorers import KNNScorer, NegBankScorer
from cmad.scorers.negbank import floor_masked
from cmad.utils import auto_device, seed_everything

TAU_QUANTILE = 0.99  # tau_val: the 99th percentile of validation-normal image scores


def _resolve_aupro(setting: object) -> bool:
    if setting == "auto":
        available = importlib.util.find_spec("anomalib") is not None
        if not available:
            warnings.warn("anomalib not installed: AUPRO is recorded as null")
        return available
    return bool(setting)


def resolve_metric(metric: str, layers: list[str], normalize: bool = False) -> str:
    """ "auto" follows the layer: raw (un-normalised) layers keep their activation magnitudes and
    are compared with Euclidean distance, normalised layers with cosine. Cosine assumes unit
    vectors, so asking for it on a raw layer is an error rather than a silent wrong number --
    unless the scorer normalises its input itself (`normalize`), as AnomalyDINO does."""
    raw = [layer.endswith("_raw") for layer in layers]
    if metric == "auto":
        if normalize:
            return "cosine"
        if any(raw) and not all(raw):
            raise ValueError(f"cannot pick a metric for mixed raw/normalised {layers}")
        return "euclidean" if all(raw) else "cosine"
    if metric == "cosine" and any(raw) and not normalize:
        raise ValueError(f"cosine needs unit vectors but {layers} contains raw layers")
    return metric


def resolve_rotations(setting: object) -> tuple[float, ...] | None:
    """None, "official8" (AnomalyDINO's 8 angles) or an explicit list of degrees."""
    if setting in (None, "", "none"):
        return None
    if setting == "official8":
        return tuple(ROTATIONS_OFFICIAL)
    return tuple(float(a) for a in setting)  # type: ignore[union-attr]


def check_roi_layer(mode: str | None, backbone: str, layer: str) -> None:
    """The official mask thresholds the first PCA component at an absolute value tuned to raw
    DINOv2 tokens. On unit-normalised tokens (or another backbone's scale) that threshold is never
    reached, every image comes out fully masked and scores 0 without any error."""
    if mode in (None, "", "none"):
        return
    if not (backbone.startswith("dinov2") and layer.endswith("_raw")):
        raise ValueError(
            f"roi={mode!r} needs raw DINOv2 tokens (a '*_raw' layer such as last_raw), "
            f"got backbone {backbone!r} with layer {layer!r}"
        )


def make_roi(mode: str | None, category: str, patches, grid, dataset: str = "mvtec_ad"):
    """The foreground mask for scoring, or None. `patches` must be raw tokens of the first layer.

    pca_official: AnomalyDINO's mask for every category; pca_official_mvtec: only for the five
    MVTec AD categories its table masks. Warns when many images come out empty or full, which
    means the mask does not fit the data.
    """
    if mode in (None, "", "none"):
        return None
    if mode == "pca_official_mvtec" and (
        dataset != "mvtec_ad" or category not in MVTEC_MASKED_CATEGORIES
    ):
        return None
    if mode not in ("pca_official", "pca_official_mvtec"):
        raise ValueError(f"unknown roi {mode!r}")
    roi = OfficialRoiMask().predict(patches, grid)
    share = roi.flatten(1).float().mean(dim=1)
    degenerate = float(((share == 0) | (share == 1)).float().mean())
    if degenerate > 0.05:
        warnings.warn(
            f"roi={mode!r} leaves {degenerate:.0%} of the {len(share)} images of {category!r} "
            "fully masked or fully foreground"
        )
    return roi


def _factors(cfg: Mapping, res: int, with_aupro: bool, split: Split) -> dict:
    """Everything that determines a run's numbers (and nothing about where it ran)."""
    scorer = {k: v for k, v in cfg["scorer"].items() if k != "name"}
    normalize = bool(scorer.pop("normalize", False))
    scorer["metric"] = resolve_metric(
        scorer["metric"], list(cfg["layers"]["names"]), normalize
    )
    if scorer.get("source") == "gt_mask":
        scorer.pop(
            "top_frac", None
        )  # gt_mask never reads it: runs differing only there are the same run
    if "lambdas" in scorer:  # 1 and 1.0 must hash alike
        scorer["lambdas"] = [float(x) for x in scorer["lambdas"]]
    # only non-default choices enter the identity, so older results keep their hashes
    extras: dict = {}
    if normalize:
        scorer["normalize"] = True
    if (cfg.get("interpolation") or "bilinear") != "bilinear":
        extras["interpolation"] = cfg["interpolation"]
    if cfg.get("roi") not in (None, "", "none"):
        extras["roi"] = cfg["roi"]
    rotations = resolve_rotations(cfg.get("reference_rotations"))
    if rotations:
        extras["reference_rotations"] = list(rotations)
    return {
        **extras,
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


@dataclass
class _Support:
    """The labelled support faults, per defect type, built from one ordered id list each.

    The negative bank is built from these images as they are: no rotated copies, even when the
    normal bank is rotated (`reference_rotations`). Rotating it would multiply the negative
    patches by the number of angles; for orientation-variable categories a defect rotated
    relative to its support image therefore sees a larger distance to the negative bank.
    """

    ids: dict[str, list[str]]
    features: dict[str, FeatureBatch]
    roi: dict[str, torch.Tensor]  # N x P bool, only when the run has an ROI
    coverage: dict[str, torch.Tensor]  # N x P float: defect area fraction of each patch


def _support_inputs(
    split: Split,
    manifest: pd.DataFrame,
    directories: dict[str, dict[str, Path]],
    origin: pd.Series,
    layers: list[str],
    roi_mode: str | None,
    dataset: str,
    raw_root: Path,
    res: int,
    resize: int | None,
) -> _Support | None:
    """Features, ROI and GT coverage of the support images. Rows, features and ids all come from
    the same ordered list per defect type, and that is asserted, because a shuffled mask would
    still have the right shape and attach to the wrong image."""
    if not split.support_abnormal:
        return None
    by_id = manifest.set_index("image_id")
    out = _Support({}, {}, {}, {})
    for defect_type, ids in sorted(split.support_abnormal.items()):
        features = _gather(directories, origin, ids, layers)
        rows = by_id.loc[ids].reset_index()
        n_features = features.patches[layers[0]].shape[0]
        if rows["image_id"].tolist() != ids or n_features != len(ids):
            raise AssertionError(
                f"support rows and features disagree for {defect_type!r}"
            )
        out.ids[defect_type] = list(ids)
        out.features[defect_type] = features
        roi = make_roi(
            roi_mode,
            split.category,
            features.patches[layers[0]],
            features.grid,
            dataset,
        )
        if roi is not None:
            out.roi[defect_type] = roi.reshape(len(ids), -1)
        out.coverage[defect_type] = mask_coverage(
            rows, raw_root, res, features.grid, resize
        )
    return out


def _selection_diagnostic(scorer: NegBankScorer, support: _Support) -> dict:
    """Precision and recall of the selected negative patches against the GT defect patches
    (patches the defect mask touches), per defect type and overall. For `gt_mask` both are 1.

    With an ROI, `top_q` and `random` can only pick foreground patches, while `recall` counts all
    defect patches, including those outside the ROI. `n_defect_in_roi` and `recall_in_roi` give
    the recall against what was selectable (null when the run has no ROI).
    """

    def ratios(counts: list[int], with_roi: bool) -> dict:
        hit, n_selected, n_defect, hit_roi, n_defect_roi = counts
        return {
            "n_selected": n_selected,
            "n_defect": n_defect,
            "precision": hit / n_selected if n_selected else None,
            "recall": hit / n_defect if n_defect else None,
            "n_defect_in_roi": n_defect_roi if with_roi else None,
            "recall_in_roi": (hit_roi / n_defect_roi if n_defect_roi else None)
            if with_roi
            else None,
        }

    per_type, totals = {}, [0] * 5
    for defect_type, selected in scorer.neg_selection.items():
        defect = coverage_to_patches(support.coverage[defect_type])
        counts = [
            int((selected & defect).sum()),
            int(selected.sum()),
            int(defect.sum()),
        ]
        roi = support.roi.get(defect_type)
        if roi is not None:
            foreground = roi.bool()
            counts += [
                int((selected & defect & foreground).sum()),
                int((defect & foreground).sum()),
            ]
        else:
            counts += [0, 0]
        per_type[defect_type] = ratios(counts, roi is not None)
        totals = [t + c for t, c in zip(totals, counts)]
    return {"per_type": per_type, "all": ratios(totals, bool(support.roi))}


def _rounded(values) -> list[float | None]:
    """Floats for the JSON record: six decimals, null for a non-finite value."""
    out = []
    for v in torch.as_tensor(values).reshape(-1).tolist():
        out.append(round(v, 6) if math.isfinite(v) else None)
    return out


def _patch_max(patch_map: torch.Tensor, roi: torch.Tensor | None) -> list[float | None]:
    """Per image maximum of an N x h x w map over the patches the scorer looks at."""
    flat = patch_map.reshape(patch_map.shape[0], -1)
    if roi is not None:
        flat = flat.masked_fill(~roi.reshape(flat.shape[0], -1).bool(), float("-inf"))
    return _rounded(flat.max(dim=1).values)


def _image_scores(rows: pd.DataFrame, out, roi: torch.Tensor | None) -> dict:
    """Per image columns of a run (test or validation), so any subset can be scored afterwards.

    `score` is the image score at the operating lambda. A negative bank adds the score at every
    lambda of the grid and the per image maxima of both distance maps, which is what a later
    look at the lambda curve or at the normal versus fault distances needs."""
    block: dict = {
        "image_id": rows["image_id"].tolist(),
        "label": rows["label"].astype(int).tolist(),
        "defect_type": rows["defect_type"].tolist(),
        "score": _rounded(out.image_score),
    }
    if rows["condition"].notna().any():
        block["condition"] = rows["condition"].tolist()
    if "d_abnormal" in out.aux:
        block["curve"] = {
            f"{lam:g}": _rounded(scores)
            for lam, scores in sorted(out.aux["curve"].items())
        }
        block["d_normal_max"] = _patch_max(out.aux["d_normal"], roi)
        block["d_abnormal_max"] = _patch_max(out.aux["d_abnormal"], roi)
    return block


def _neg_bank_record(
    scorer: NegBankScorer,
    support: _Support | None,
    out,
    labels: np.ndarray,
    selection: dict | None,
) -> dict:
    """The `neg_bank` block of a run record: what the bank holds, the lambda curve, the oracle."""
    record: dict = {
        "source": scorer.source,
        "top_frac": None if scorer.source == "gt_mask" else scorer.top_frac,
        "lambda": scorer.lam,
        "lambdas": list(scorer.lambdas),
        # set only when leave-one-support-out chose the operating lambda; a fixed `lam` leaves it null
        "selected_lambda": scorer.lam if selection and selection["selected"] else None,
        "selection_rule": selection,
        "active": scorer.neg is not None,
    }
    if scorer.neg is None or support is None:
        warnings.warn("negbank run without any support image: this is plain kNN")
        return {**record, "n_negative_patches": 0, "curve": None, "oracle": None}
    curve = {}
    for lam, scores in sorted(out.aux["curve"].items()):
        m = image_metrics(scores.numpy(), labels)
        curve[f"{lam:g}"] = {
            "image_auroc": m["image_auroc"],
            "tpr_at_1pct_fpr": m["tpr_at_1pct_fpr"],
        }
    # the best lambda on the test labels; ties go to the smaller lambda
    best = max(curve, key=lambda lam: (curve[lam]["image_auroc"], -float(lam)))
    type_of = {i: t for t, ids in support.ids.items() for i in ids}
    if record["selected_lambda"] is not None:
        # what the selection achieved on the test images, and how far it is from the oracle
        chosen = curve[f"{scorer.lam:g}"]["image_auroc"]
        record["selected_image_auroc"] = chosen
        record["oracle_gap"] = curve[best]["image_auroc"] - chosen
    return {
        **record,
        "n_negative_patches": int(scorer.neg.shape[0]),
        "bytes": int(scorer.neg.numel() * scorer.neg.element_size()),
        "support": {
            image_id: {"defect_type": type_of[image_id], "n_negative_patches": n}
            for image_id, n in scorer.neg_counts.items()
        },
        "curve": curve,
        "oracle": {  # chosen with the test labels: an upper bound, not deployable
            "lambda": float(best),
            "image_auroc": curve[best]["image_auroc"],
        },
        "selection": _selection_diagnostic(scorer, support),
    }


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
    is_negbank = cfg["scorer"]["name"] == "negbank"
    if is_negbank and spec.n_abnormal_per_type is None:
        raise ValueError(
            f"scorer negbank needs a regime with labelled support faults (few_abn_k<k>), got {spec.name!r}"
        )

    factors = _factors(cfg, res, with_aupro, split)
    check_roi_layer(cfg.get("roi"), backbone, factors["layers"][0])
    digest = config_hash(factors)
    results_dir = data_root / "results" / "runs"

    path = run_path(results_dir, digest)
    if path.exists() and not cfg["force"]:
        return json.loads(path.read_text()), False

    start = time.perf_counter()
    seed_everything(seed)
    device = auto_device() if cfg["device"] == "auto" else cfg["device"]
    preprocess = PreprocessConfig(
        res=res, resize=resize, interpolation=cfg.get("interpolation") or "bilinear"
    )
    rotations = resolve_rotations(cfg.get("reference_rotations"))
    layers = factors["layers"]

    support_ids = [i for ids in split.support_abnormal.values() for i in ids]
    origin = manifest.set_index("image_id")["orig_split"]
    # rotated references are extracted on the fly, so the cached train shard is not needed then
    cached_ids = split.val_normal + support_ids + split.test
    if not rotations:
        cached_ids = split.train_normal + cached_ids
    needed = sorted(set(origin[cached_ids]))
    directories = {
        o: _shard_dirs(data_root, manifest, category, o, backbone, layers, preprocess)
        for o in needed
    }

    # compute on miss: the extractor (and its weights) is only built if a shard is absent
    missing = [o for o in needed if not all(map(Shard.exists, directories[o].values()))]
    extract_s = 0.0
    built: list[FeatureExtractor] = []

    def get_extractor() -> FeatureExtractor:
        if (
            not built
        ):  # the backbone (and its weights) is only built when something needs it
            make = extractor_factory or (lambda d: build_extractor(backbone, device=d))
            built.append(make(device))
        return built[0]

    if missing:
        t = time.perf_counter()
        extractor = get_extractor()
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

    if rotations:
        t = time.perf_counter()
        train_rows = (
            manifest.set_index("image_id").loc[split.train_normal].reset_index()
        )
        train = extract_rotated(
            get_extractor(),
            train_rows,
            data_root / "raw",
            preprocess,
            layers,
            rotations,
        )
        extract_s += time.perf_counter() - t
    else:
        train = _gather(directories, origin, split.train_normal, layers)
    test = _gather(directories, origin, split.test, layers)
    n_train_patches = train.patches[layers[0]].shape[0] * train.grid[0] * train.grid[1]
    roi_mode = cfg.get("roi")
    # the mask is computed from the raw tokens of the first layer; the bank is never masked
    test_roi = make_roi(
        roi_mode, category, test.patches[layers[0]], test.grid, cfg["dataset"]
    )

    sc = factors["scorer_params"]
    common = {
        "k": sc["k"],
        "coreset_fraction": sc["coreset_fraction"],
        "coreset_proj_dim": sc["coreset_proj_dim"],
        "image_agg": sc["image_agg"],
        "metric": sc["metric"],
        "device": device,
        "seed": seed,
        "normalize": sc.get("normalize", False),
    }
    support = None
    t = time.perf_counter()
    if is_negbank:
        scorer = NegBankScorer(
            lambdas=tuple(sc["lambdas"]),
            source=sc["source"],
            top_frac=sc.get("top_frac", 0.10),
            **common,
        )
        support = _support_inputs(
            split,
            manifest,
            directories,
            origin,
            layers,
            roi_mode,
            cfg["dataset"],
            data_root / "raw",
            res,
            resize,
        )
        scorer.fit(
            train,
            support=support.features if support else None,
            support_masks={
                t_: coverage_to_patches(c) for t_, c in support.coverage.items()
            }
            if support
            else None,
            support_ids=support.ids if support else None,
            support_roi=support.roi if support else None,
        )
    else:
        scorer = KNNScorer(**common)
        scorer.fit(train)  # support abnormals are only used by the negative bank
    fit_s = time.perf_counter() - t

    val = val_roi = None
    if split.val_normal:
        val = _gather(directories, origin, split.val_normal, layers)
        val_roi = make_roi(
            roi_mode, category, val.patches[layers[0]], val.grid, cfg["dataset"]
        )

    selection = None
    if is_negbank:
        if sc["lam"] == "select":
            # leave-one-support-out on val_normal + support: no labelled test image is involved
            t = time.perf_counter()
            selection = scorer.select_lambda(val, val_roi)
            selection["seconds"] = time.perf_counter() - t
        else:
            scorer.lam = float(sc["lam"])  # a fixed operating point

    t = time.perf_counter()
    out = scorer.score(test, roi=test_roi)
    if is_negbank and test_roi is not None:
        # masked patches share one floor over all test images, so pooled pixel metrics cannot
        # rank a masked background patch of one image above valid patches of another
        out.patch_map = floor_masked(out.patch_map, test_roi)
    tau = val_out = None
    if val is not None:
        val_out = scorer.score(val, roi=val_roi)
        tau = float(np.quantile(val_out.image_score.numpy(), TAU_QUANTILE))
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
        "image_scores": {
            "test": _image_scores(test_rows, out, test_roi),
            "val": None
            if val_out is None
            else _image_scores(
                manifest.set_index("image_id").loc[split.val_normal].reset_index(),
                val_out,
                val_roi,
            ),
        },
        **(
            {
                "neg_bank": _neg_bank_record(
                    scorer, support, out, test_rows["label"].to_numpy(), selection
                )
            }
            if is_negbank
            else {}
        ),
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
