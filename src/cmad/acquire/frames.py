from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from PIL import Image


@dataclass
class Frame:
    image: np.ndarray  # HxWx3 uint8 RGB
    meta: dict = field(default_factory=dict)


@dataclass
class PreprocessedFrame:
    tensor: torch.Tensor  # 3xHxW, ImageNet-normalised
    roi_mask: torch.Tensor | None  # 1 x h x w at patch-grid resolution
    keep: bool = True


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


META_COLUMNS = ("image_id", "category", "label", "defect_type")


class ManifestSource:
    def __init__(self, manifest: pd.DataFrame, raw_root: Path) -> None:
        self.manifest = manifest
        self.raw_root = Path(raw_root)

    def __len__(self) -> int:
        return len(self.manifest)

    def __iter__(self) -> Iterator[Frame]:
        for row in self.manifest.itertuples(index=False):
            # convert("RGB") also lifts MVTec's grayscale categories to 3 channels
            with Image.open(self.raw_root / row.image_path) as img:
                image = np.asarray(img.convert("RGB"))
            yield Frame(image=image, meta={c: getattr(row, c) for c in META_COLUMNS})
