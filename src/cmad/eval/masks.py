from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

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
