from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from cmad.acquire import resize_crop


def load_masks(
    rows: pd.DataFrame, raw_root: Path, res: int, resize: int | None = None
) -> torch.Tensor:
    """N x res x res bool masks, cut with exactly the geometry the images got (nearest-neighbour)."""
    geometry = resize_crop(res, resize, "nearest")
    out = torch.zeros(len(rows), res, res, dtype=torch.bool)
    for i, row in enumerate(rows.itertuples(index=False)):
        if pd.isna(row.mask_path):
            if row.label == 1:
                raise ValueError(f"anomalous image without a mask: {row.image_id}")
            continue
        with Image.open(Path(raw_root) / row.mask_path) as m:
            mask = torch.from_numpy(np.array(m.convert("L")) > 0)
        out[i] = geometry(mask[None].to(torch.uint8))[0] > 0
    return out


def _resized_size(h: int, w: int, short: int) -> tuple[int, int]:
    """Size torchvision's Resize(short) gives an h x w image (long side truncated, as there)."""
    if h <= w:
        return short, int(short * w / h)
    return int(short * h / w), short


def mask_coverage(
    rows: pd.DataFrame,
    raw_root: Path,
    res: int,
    grid: tuple[int, int],
    resize: int | None = None,
) -> torch.Tensor:
    """N x P float: the fraction of each patch covered by the defect (row-major, like the features).

    The mask gets the image's geometry (short side -> `resize`, centre crop to `res`) but is
    resampled by *area*, not nearest-neighbour: nearest keeps every 2nd-3rd pixel of a 1024 px mask,
    so a defect that only clips a cell can vanish from it, and which cells it vanishes from depends
    on the sampling phase. Area resampling loses no source pixel, so "any overlap" means any overlap.
    Rows without a mask (normal images) get zeros; an anomalous row without one raises.
    """
    gh, gw = grid
    if res % gh or res % gw:
        raise ValueError(
            f"a {gh}x{gw} grid does not divide the {res} px image: the mask and the features "
            "were made at different resolutions"
        )
    crop = v2.CenterCrop(res)
    out = torch.zeros(len(rows), gh * gw)
    for i, row in enumerate(rows.itertuples(index=False)):
        if pd.isna(row.mask_path):
            if row.label == 1:
                raise ValueError(f"anomalous image without a mask: {row.image_id}")
            continue
        with Image.open(Path(raw_root) / row.mask_path) as m:
            mask = torch.from_numpy(np.array(m.convert("L")) > 0).float()
        size = _resized_size(*mask.shape, resize or res)
        resized = F.interpolate(mask[None, None], size=size, mode="area")[0]
        cell = F.adaptive_avg_pool2d(crop(resized)[None], (gh, gw))
        out[i] = cell.reshape(-1)
    return out


def coverage_to_patches(
    coverage: torch.Tensor, min_coverage: float = 0.0
) -> torch.Tensor:
    """N x P coverage -> N x P bool: patches covered by more than `min_coverage` of their area.

    The default 0 means any overlapping pixel counts. Whatever the threshold, the
    best-covered patch of a non-empty mask is always kept, so a defect smaller than the threshold
    (or than a patch) still contributes. An empty mask (defect cropped away) stays empty.
    """
    sel = coverage > min_coverage
    best = coverage.argmax(dim=1)
    nonempty = coverage.max(dim=1).values > 0
    sel[nonempty, best[nonempty]] = True
    return sel


def support_patch_masks(
    rows: pd.DataFrame,
    raw_root: Path,
    res: int,
    grid: tuple[int, int],
    resize: int | None = None,
    min_coverage: float = 0.0,
) -> torch.Tensor:
    """N x P bool `gt_mask` negatives of the support rows, in the order of `rows`."""
    return coverage_to_patches(
        mask_coverage(rows, raw_root, res, grid, resize), min_coverage
    )
