"""Manifest adapter for MVTec AD 2, public split only.

Layout on disk: raw/mvtec_ad2/<category>/{train,validation}/good/<id>_<condition>.png
and .../test_public/{good,bad}/<id>_<condition>.png, with bad-image masks under
.../test_public/ground_truth/bad/<id>_<condition>_mask.png.

`test_private` and `test_private_mixed` are server-evaluated splits with no
public ground truth, and are never globbed here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import ManifestRow, make_row, rows_to_dataframe

DATASET = "mvtec_ad2"


def build_manifest(raw_root: Path) -> pd.DataFrame:
    dataset_root = raw_root / DATASET
    rows: list[ManifestRow] = []
    for category_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        rows.extend(_ingest_category(raw_root, category_dir.name, category_dir))
    return rows_to_dataframe(rows)


def _condition_of(image_path: Path) -> str:
    _id, _sep, condition = image_path.stem.partition("_")
    return condition


def _ingest_split(
    raw_root: Path,
    category: str,
    split_dir: Path,
    orig_split: str,
    label: int,
    defect_type: str,
    mask_dir: Path | None = None,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    if not split_dir.is_dir():
        return rows
    for img_path in sorted(split_dir.glob("*.png")):
        mask_path = None
        if mask_dir is not None:
            candidate = mask_dir / f"{img_path.stem}_mask.png"
            mask_path = candidate if candidate.exists() else None
        rows.append(
            make_row(
                raw_root=raw_root,
                image_path=img_path,
                dataset=DATASET,
                category=category,
                orig_split=orig_split,
                label=label,
                defect_type=defect_type,
                anomaly_kind="none" if label == 0 else "structural",
                condition=_condition_of(img_path),
                mask_path=mask_path,
            )
        )
    return rows


def _ingest_category(raw_root: Path, category: str, category_dir: Path) -> list[ManifestRow]:
    test_public = category_dir / "test_public"
    rows: list[ManifestRow] = []
    rows += _ingest_split(raw_root, category, category_dir / "train" / "good", "train", 0, "good")
    rows += _ingest_split(raw_root, category, category_dir / "validation" / "good", "validation", 0, "good")
    rows += _ingest_split(raw_root, category, test_public / "good", "test_public", 0, "good")
    rows += _ingest_split(
        raw_root,
        category,
        test_public / "bad",
        "test_public",
        1,
        "bad",
        mask_dir=test_public / "ground_truth" / "bad",
    )
    return rows
