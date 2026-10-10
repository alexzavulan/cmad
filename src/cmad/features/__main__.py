"""CLI: python -m cmad.features extract --backbone dinov2_vits14 --layers last mid --tier dev"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from cmad.acquire import PreprocessConfig

from .base import build_extractor, default_resolution
from .cache import extract_to_cache

DEFAULT_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cmad.features")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="extract features for a tier into the cache")
    ex.add_argument("--backbone", required=True)
    ex.add_argument("--layers", nargs="+", required=True)
    ex.add_argument("--tier", default="dev")
    ex.add_argument("--res", type=int, default=None, help="default: the backbone's own")
    ex.add_argument(
        "--resize",
        type=int,
        default=None,
        help="short-side size before the centre crop to --res (PatchCore: 256 with --res 224)",
    )
    ex.add_argument("--device", default="auto")
    ex.add_argument("--batch-size", type=int, default=16)
    ex.add_argument(
        "--data-root", type=Path, default=None, help="overrides $CMAD_DATA_ROOT"
    )
    ex.add_argument("--configs-dir", type=Path, default=DEFAULT_CONFIGS)
    args = parser.parse_args(argv)

    if args.data_root is not None:
        data_root = args.data_root
    elif "CMAD_DATA_ROOT" in os.environ:
        data_root = Path(os.environ["CMAD_DATA_ROOT"]).expanduser()
    else:
        parser.error("CMAD_DATA_ROOT is not set; export it or pass --data-root")
        return

    tier = OmegaConf.load(args.configs_dir / "tier" / f"{args.tier}.yaml")
    manifest = pd.read_parquet(data_root / "manifests" / f"{tier.dataset}.parquet")
    res = args.res if args.res is not None else default_resolution(args.backbone)
    device = _auto_device() if args.device == "auto" else args.device
    extractor = build_extractor(args.backbone, device=device)
    preprocess = PreprocessConfig(res=res, resize=args.resize)

    print(
        f"{args.backbone} layers={args.layers} res={res} resize={args.resize or res} "
        f"device={device} tier={args.tier}"
    )
    for category in tier.categories:
        splits = sorted(
            manifest.loc[manifest["category"] == category, "orig_split"].unique()
        )
        for split in splits:
            start = time.perf_counter()
            result = extract_to_cache(
                manifest=manifest,
                raw_root=data_root / "raw",
                features_dir=data_root / "features",
                extractor=extractor,
                backbone=args.backbone,
                layers=args.layers,
                preprocess=preprocess,
                category=category,
                split=split,
                batch_size=args.batch_size,
            )
            status = (
                f"extracted {list(result.extracted)}"
                if result.extracted
                else "cache hit"
            )
            print(
                f"  {category}/{split}: {status} ({time.perf_counter() - start:.1f}s)"
            )


if __name__ == "__main__":
    main()
