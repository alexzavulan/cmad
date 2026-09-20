"""Manifest adapter for MVTec AD.

Layout on disk: raw/mvtec_ad/<category>/{train,test}/<good|defect>/<id>.png,
with per-defect masks under raw/mvtec_ad/<category>/ground_truth/<defect>/<id>_mask.png.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import ManifestRow, make_row, rows_to_dataframe

DATASET = "mvtec_ad"


def build_manifest(raw_root: Path) -> pd.DataFrame:
    dataset_root = raw_root / DATASET
    rows: list[ManifestRow] = []
    for category_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        rows.extend(_ingest_category(raw_root, category_dir.name, category_dir))
    return rows_to_dataframe(rows)


def _ingest_category(raw_root: Path, category: str, category_dir: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []

    train_good = category_dir / "train" / "good"
    if train_good.is_dir():
        for img_path in sorted(train_good.glob("*.png")):
            rows.append(
                make_row(
                    raw_root=raw_root,
                    image_path=img_path,
                    dataset=DATASET,
                    category=category,
                    orig_split="train",
                    label=0,
                    defect_type="good",
                    anomaly_kind="none",
                    condition=None,
                    mask_path=None,
                )
            )

    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"
    if test_dir.is_dir():
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            is_good = defect_type == "good"
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if not is_good:
                    candidate = ground_truth_dir / defect_type / f"{img_path.stem}_mask.png"
                    mask_path = candidate if candidate.exists() else None
                rows.append(
                    make_row(
                        raw_root=raw_root,
                        image_path=img_path,
                        dataset=DATASET,
                        category=category,
                        orig_split="test",
                        label=0 if is_good else 1,
                        defect_type=defect_type,
                        anomaly_kind="none" if is_good else "structural",
                        condition=None,
                        mask_path=mask_path,
                    )
                )

    return rows
