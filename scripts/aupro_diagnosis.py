"""Why is our AUPRO below the published PatchCore PRO?  One scoring pass per category, then
AUPRO evaluated several ways on the *same* anomaly maps:

  anomalib@L      anomalib's AUPRO at FPR limits L
  exact4@L        independent exact PRO (MVTec definition), 4-connected regions
  exact8@0.3      same, 8-connected regions
  bil>0@0.3       masks interpolated bilinearly (as PIL does in PatchCore's loader), then > 0
  bil>.5@0.3      ... then > 0.5

python scripts/aupro_diagnosis.py --categories bottle tile --backbone wrn50_in1k \
    --layers cnn_l23_raw --metric euclidean --coreset 0.1 --resize 256 --res 224
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

from cmad.acquire import PreprocessConfig, resize_crop
from cmad.eval import aupro, load_masks, upsample_and_blur
from cmad.features import Shard, category_rows, shard_directories
from cmad.report import load_reference
from cmad.run import _load
from cmad.scorers import KNNScorer
from cmad.utils import auto_device, seed_everything

LIMITS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def exact_pro_curve(
    maps: np.ndarray, masks: np.ndarray, connectivity: int = 4, n_points: int = 20_001
) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, pro) with the MVTec definition: FPR over all non-anomalous pixels of the set,
    PRO = mean over connected GT regions of the fraction of the region above the threshold.
    Thresholds are the normal-pixel score quantiles, so the FPR axis is covered uniformly."""
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    normal = np.sort(maps[~masks])
    regions = []
    for m, a in zip(masks, maps, strict=True):
        if not m.any():
            continue
        labels, n = ndimage.label(m, structure=structure)
        for k in range(1, n + 1):
            regions.append(np.sort(a[labels == k]))
    idx = np.unique(np.linspace(0, len(normal) - 1, n_points).astype(np.int64))
    thr = np.concatenate([normal[idx][::-1], [normal[0] - 1.0]])  # high -> low
    thr = np.concatenate([[maps.max() + 1.0], thr])  # fpr = 0, pro = 0 anchor
    fpr = (len(normal) - np.searchsorted(normal, thr, side="right")) / len(normal)
    pro = np.zeros_like(thr)
    for r in regions:
        pro += (len(r) - np.searchsorted(r, thr, side="right")) / len(r)
    return fpr, pro / len(regions)


def area_to(fpr: np.ndarray, pro: np.ndarray, limit: float) -> float:
    keep = fpr <= limit
    x, y = fpr[keep], pro[keep]
    if x[-1] < limit and keep.sum() < len(fpr):  # interpolate onto the limit exactly
        j = keep.sum()
        t = (limit - fpr[j - 1]) / (fpr[j] - fpr[j - 1])
        x = np.append(x, limit)
        y = np.append(y, pro[j - 1] + t * (pro[j] - pro[j - 1]))
    return float(np.trapezoid(y, x) / limit)


def bilinear_masks(
    rows: pd.DataFrame, raw_root: Path, res: int, resize: int | None
) -> torch.Tensor:
    """Float masks in [0, 1] with the image's own (bilinear) geometry."""
    geometry = resize_crop(res, resize, "bilinear")
    out = torch.zeros(len(rows), res, res)
    for i, row in enumerate(rows.itertuples(index=False)):
        if pd.isna(row.mask_path):
            continue
        with Image.open(Path(raw_root) / row.mask_path) as m:
            arr = torch.from_numpy((np.array(m.convert("L")) > 0).astype(np.float32))
        out[i] = geometry(arr[None])[0]
    return out


def diagnose(
    category: str, args: argparse.Namespace, data_root: Path, ref: dict
) -> dict:
    seed_everything(0)
    device = auto_device() if args.device == "auto" else args.device
    manifest = pd.read_parquet(data_root / "manifests" / f"{args.dataset}.parquet")
    rows = {s: category_rows(manifest, category, s) for s in ("train", "test")}
    pre = PreprocessConfig(res=args.res, resize=args.resize)
    dirs = {
        s: shard_directories(
            features_dir=data_root / "features",
            backbone=args.backbone,
            layers=args.layers,
            preprocess=pre,
            category=category,
            split=s,
            image_ids=tuple(rows[s]["image_id"]),
        )
        for s in rows
    }
    for s, layer_dirs in dirs.items():
        if not all(map(Shard.exists, layer_dirs.values())):
            raise SystemExit(f"features missing for {category}/{s}; run cmad.run first")

    scorer = KNNScorer(
        metric=args.metric, coreset_fraction=args.coreset, device=device, seed=0
    )
    scorer.fit(_load(dirs["train"]))
    out = scorer.score(_load(dirs["test"]))
    maps = upsample_and_blur(out.patch_map, size=args.res, sigma=4.0)
    raw = data_root / "raw"
    masks = load_masks(rows["test"], raw, args.res, args.resize)

    res: dict = {"category": category}
    try:
        for lim in LIMITS:
            res[f"anomalib@{lim:g}"] = 100 * aupro(maps, masks, lim)
    except ImportError:
        pass
    m_np, a_np = masks.numpy(), maps.numpy()
    for conn in (4, 8):
        fpr, pro = exact_pro_curve(a_np, m_np, conn)
        for lim in LIMITS if conn == 4 else (0.3,):
            res[f"exact{conn}@{lim:g}"] = 100 * area_to(fpr, pro, lim)
    bil = bilinear_masks(rows["test"], raw, args.res, args.resize).numpy()
    for name, thresh in (("bil>0", 0.0), ("bil>.5", 0.5)):
        fpr, pro = exact_pro_curve(a_np, bil > thresh)
        res[f"{name}@0.3"] = 100 * area_to(fpr, pro, 0.3)
    res["ref_pro"] = ref["per_category"].get(category)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--dataset", default="mvtec_ad")
    p.add_argument("--backbone", default="wrn50_in1k")
    p.add_argument("--layers", nargs="+", default=["cnn_l23_raw"])
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--coreset", type=float, default=0.1)
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--resize", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    data_root = Path(os.environ["CMAD_DATA_ROOT"]).expanduser()
    ref = load_reference(args.coreset)["aupro_0.3"]

    results = []
    for category in args.categories:
        t = time.perf_counter()
        results.append(diagnose(category, args, data_root, ref))
        print(f"{category}: {time.perf_counter() - t:.0f}s", flush=True)
    df = pd.DataFrame(results).set_index("category")
    df.loc["MEAN"] = df.mean()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.round(1).to_string())
    out = args.out or data_root / "results" / "diagnostics" / "aupro_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(df.to_dict("index"), indent=2))


if __name__ == "__main__":
    main()
