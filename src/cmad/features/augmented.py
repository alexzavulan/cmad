"""Features of rotated reference images, computed on the fly (a few-shot bank is tiny)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import torch

from cmad.acquire import (
    Frame,
    ManifestSource,
    PreprocessConfig,
    Preprocessor,
    rotate_image,
)

from .base import FeatureBatch, FeatureExtractor


def _rotated_tensors(
    rows: pd.DataFrame,
    raw_root: Path,
    preprocess: PreprocessConfig,
    angles: Sequence[float],
) -> Iterator[torch.Tensor]:
    prep = Preprocessor(preprocess)
    for frame in ManifestSource(rows, raw_root):
        for angle in angles:
            # rotate the full-resolution image first, then the usual resize + normalise
            yield prep(
                Frame(image=rotate_image(frame.image, angle), meta=frame.meta)
            ).tensor


def extract_rotated(
    extractor: FeatureExtractor,
    rows: pd.DataFrame,
    raw_root: Path,
    preprocess: PreprocessConfig,
    layers: list[str],
    angles: Sequence[float],
    batch_size: int = 16,
) -> FeatureBatch:
    """Features of every row at every angle, image-major: (image 0, all angles), (image 1, ...)."""
    chunks: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
    globals_: list[torch.Tensor] = []
    grid: tuple[int, int] | None = None
    batch: list[torch.Tensor] = []

    def flush() -> None:
        nonlocal grid
        out = extractor.extract(torch.stack(batch), layers)
        grid = out.grid
        for layer in layers:
            chunks[layer].append(out.patches[layer].cpu())
        globals_.append(out.global_token.cpu())
        batch.clear()

    for tensor in _rotated_tensors(rows, raw_root, preprocess, angles):
        batch.append(tensor)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    if grid is None:
        raise ValueError("no reference images to extract")
    return FeatureBatch(
        patches={layer: torch.cat(parts) for layer, parts in chunks.items()},
        grid=grid,
        global_token=torch.cat(globals_),
    )
