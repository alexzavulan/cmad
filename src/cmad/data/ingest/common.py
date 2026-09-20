"""Shared helpers for per-dataset manifest adapters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "category",
    "orig_split",
    "image_path",
    "mask_path",
    "label",
    "defect_type",
    "anomaly_kind",
    "condition",
    "width",
    "height",
    "sha1",
]


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    dataset: str
    category: str
    orig_split: str
    image_path: str
    mask_path: str | None
    label: int
    defect_type: str
    anomaly_kind: str
    condition: str | None
    width: int
    height: int
    sha1: str


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def make_row(
    *,
    raw_root: Path,
    image_path: Path,
    dataset: str,
    category: str,
    orig_split: str,
    label: int,
    defect_type: str,
    anomaly_kind: str,
    condition: str | None,
    mask_path: Path | None,
) -> ManifestRow:
    rel_image_path = image_path.relative_to(raw_root).as_posix()
    rel_mask_path = mask_path.relative_to(raw_root).as_posix() if mask_path else None
    width, height = image_size(image_path)
    return ManifestRow(
        image_id=rel_image_path,
        dataset=dataset,
        category=category,
        orig_split=orig_split,
        image_path=rel_image_path,
        mask_path=rel_mask_path,
        label=label,
        defect_type=defect_type,
        anomaly_kind=anomaly_kind,
        condition=condition,
        width=width,
        height=height,
        sha1=sha1_of_file(image_path),
    )


def rows_to_dataframe(rows: list[ManifestRow]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(row) for row in rows], columns=MANIFEST_COLUMNS)
    return df


def write_manifest(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
