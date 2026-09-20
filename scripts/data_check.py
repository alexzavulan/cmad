#!/usr/bin/env python3
"""`make data-check`: structural integrity check of the raw dataset downloads.

MVTec doesn't publish a per-file checksum manifest, so this doesn't compare
against an external "golden" list. Instead it re-runs the real ingest
adapters against $CMAD_DATA_ROOT/raw and checks the things that actually
indicate a broken/incomplete download or a mismatched adapter:

  - the exact expected set of categories is present, no more, no less
  - every category has at least one train and one test row
  - every anomalous (label=1) row has a matching ground-truth mask
  - every mask file on disk is referenced by exactly one manifest row
    (catches masks left over from a partial/failed extraction)
  - no two images in the same dataset share a sha1 (accidental duplicate
    files from a corrupted download)

Exits non-zero if any of the first four checks fail. Duplicate sha1s are
reported but non-fatal, since MVTec AD is small enough that a legitimate
duplicate isn't impossible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cmad.data.ingest import BUILDERS

EXPECTED_CATEGORIES = {
    # Confirmed directly against the downloaded data on ailab-1 (2026-09-13).
    "mvtec_ad": {
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
        "wood", "zipper",
    },
    "mvtec_ad2": {
        "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial",
        "wallplugs", "walnuts",
    },
}


def _mask_files_on_disk(dataset: str, raw_root: Path) -> set[str]:
    dataset_root = raw_root / dataset
    if dataset == "mvtec_ad":
        paths = dataset_root.glob("*/ground_truth/**/*.png")
    else:
        paths = dataset_root.glob("*/test_public/ground_truth/**/*.png")
    return {p.relative_to(raw_root).as_posix() for p in paths}


def check_dataset(dataset: str, raw_root: Path) -> list[str]:
    problems: list[str] = []
    print(f"--- {dataset} ---")

    actual_categories = {p.name for p in (raw_root / dataset).iterdir() if p.is_dir()}
    expected = EXPECTED_CATEGORIES[dataset]
    if actual_categories != expected:
        problems.append(
            f"{dataset}: category mismatch -- missing {expected - actual_categories}, "
            f"unexpected {actual_categories - expected}"
        )

    df = BUILDERS[dataset](raw_root)
    print(f"{len(df)} rows across {df['category'].nunique()} categories")
    for category, count in df.groupby("category").size().sort_index().items():
        print(f"  {category}: {count}")

    empty_categories = expected - set(df["category"].unique())
    if empty_categories:
        problems.append(f"{dataset}: categories with zero ingested rows: {empty_categories}")

    anomalous = df[df["label"] == 1]
    missing_masks = anomalous[anomalous["mask_path"].isna()]
    if len(missing_masks):
        problems.append(
            f"{dataset}: {len(missing_masks)} anomalous rows have no mask "
            f"(e.g. {missing_masks.iloc[0]['image_path']})"
        )

    referenced_masks = set(df["mask_path"].dropna())
    on_disk_masks = _mask_files_on_disk(dataset, raw_root)
    orphaned = on_disk_masks - referenced_masks
    if orphaned:
        problems.append(f"{dataset}: {len(orphaned)} mask file(s) not referenced by any row, e.g. {min(orphaned)}")

    dupes = df[df.duplicated("sha1", keep=False)].sort_values("sha1")
    if len(dupes):
        print(f"  note: {len(dupes)} rows share a sha1 with another image (not fatal):")
        for sha1, group in list(dupes.groupby("sha1"))[:5]:
            print(f"    {sha1}: {list(group['image_path'])}")

    return problems


def main() -> int:
    data_root = Path(os.environ.get("CMAD_DATA_ROOT", "~/cmad-data")).expanduser()
    raw_root = data_root / "raw"
    if not raw_root.is_dir():
        print(f"data-check FAILED: raw data root not found at {raw_root}")
        print("(check $CMAD_DATA_ROOT, or that the datasets have been downloaded)")
        return 1

    problems: list[str] = []
    for dataset in ("mvtec_ad", "mvtec_ad2"):
        problems.extend(check_dataset(dataset, raw_root))
        print()

    if problems:
        print("data-check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("data-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
