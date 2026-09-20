"""CLI: python -m cmad.data.ingest mvtec_ad mvtec_ad2"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import BUILDERS
from .common import write_manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build per-dataset image manifests.")
    parser.add_argument("datasets", nargs="+", choices=sorted(BUILDERS))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="overrides $CMAD_DATA_ROOT",
    )
    args = parser.parse_args(argv)

    if args.data_root is not None:
        data_root = args.data_root
    elif "CMAD_DATA_ROOT" in os.environ:
        data_root = Path(os.environ["CMAD_DATA_ROOT"]).expanduser()
    else:
        parser.error("CMAD_DATA_ROOT is not set; export it or pass --data-root")
        return

    raw_root = data_root / "raw"
    manifests_dir = data_root / "manifests"

    for dataset in args.datasets:
        df = BUILDERS[dataset](raw_root)
        out_path = manifests_dir / f"{dataset}.parquet"
        write_manifest(df, out_path)
        print(f"{dataset}: {len(df)} rows across {df['category'].nunique()} categories -> {out_path}")
        for category, count in df.groupby("category").size().sort_index().items():
            print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
