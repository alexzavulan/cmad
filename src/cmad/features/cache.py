from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cmad.acquire import ManifestSource, PreprocessConfig, Preprocessor

from .base import FeatureExtractor

# bump whenever extractor output changes for the same inputs, so stale shards are never reused
SCHEMA_VERSION = 1

SHARD_FILE = "shard.safetensors"
INDEX_FILE = "index.json"


@dataclass(frozen=True)
class CacheKey:
    backbone: str
    layer: str
    res: int
    preproc_digest: str
    category: str
    split: str
    image_ids: tuple[str, ...]
    corruption: str = "none"

    def digest(self) -> str:
        payload = json.dumps(
            {**asdict(self), "schema_version": SCHEMA_VERSION}, sort_keys=True
        )
        return hashlib.sha1(payload.encode()).hexdigest()

    def directory(self, features_dir: Path) -> Path:
        return (
            features_dir
            / self.backbone
            / self.layer
            / self.category
            / self.split
            / self.digest()[:12]
        )


class Shard:
    """Read side of one cached (backbone, layer, category, split) shard."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index = json.loads((directory / INDEX_FILE).read_text())
        self.image_ids: list[str] = self.index["image_ids"]
        self.grid: tuple[int, int] = tuple(self.index["grid"])
        self._row = {image_id: row for row, image_id in enumerate(self.image_ids)}

    @staticmethod
    def exists(directory: Path) -> bool:
        # index.json is renamed into place last, so its presence marks a complete shard
        return (directory / INDEX_FILE).exists()

    def _read(self, name: str, image_ids: list[str] | None) -> torch.Tensor:
        with safe_open(self.directory / SHARD_FILE, framework="pt") as f:
            tensor = f.get_tensor(name)
        if image_ids is None:
            return tensor
        return tensor[[self._row[i] for i in image_ids]]

    def patches(self, image_ids: list[str] | None = None) -> torch.Tensor:
        return self._read("patches", image_ids)

    def global_token(self, image_ids: list[str] | None = None) -> torch.Tensor:
        return self._read("global", image_ids)


def _write_shard(
    directory: Path, patches: torch.Tensor, global_token: torch.Tensor, index: dict
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp_shard = directory / f"{SHARD_FILE}.tmp"
    save_file(
        {"patches": patches.contiguous(), "global": global_token.contiguous()},
        tmp_shard,
    )
    os.replace(tmp_shard, directory / SHARD_FILE)
    tmp_index = directory / f"{INDEX_FILE}.tmp"
    tmp_index.write_text(json.dumps(index))
    os.replace(tmp_index, directory / INDEX_FILE)


def _batches(
    source: ManifestSource, preprocessor: Preprocessor, batch_size: int
) -> Iterator[torch.Tensor]:
    batch: list[torch.Tensor] = []
    for frame in source:
        batch.append(preprocessor(frame).tensor)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    if batch:
        yield torch.stack(batch)


@dataclass(frozen=True)
class CacheResult:
    directories: dict[str, Path]  # layer -> shard directory
    extracted: tuple[str, ...]  # layers computed this call (empty = pure cache hit)


def category_rows(manifest: pd.DataFrame, category: str, split: str) -> pd.DataFrame:
    """Manifest rows of one (category, split) in the cache's image order (sorted by image_id)."""
    rows = manifest[
        (manifest["category"] == category) & (manifest["orig_split"] == split)
    ].sort_values("image_id")
    if rows.empty:
        raise ValueError(f"no manifest rows for category={category!r} split={split!r}")
    return rows


def shard_directories(
    *,
    features_dir: Path,
    backbone: str,
    layers: list[str],
    preprocess: PreprocessConfig,
    category: str,
    split: str,
    image_ids: tuple[str, ...],
    corruption: str = "none",
) -> dict[str, Path]:
    """layer -> shard directory, whether or not the shard exists yet."""
    return {
        layer: CacheKey(
            backbone=backbone,
            layer=layer,
            res=preprocess.res,
            preproc_digest=preprocess.digest(),
            category=category,
            split=split,
            image_ids=image_ids,
            corruption=corruption,
        ).directory(features_dir)
        for layer in dict.fromkeys(layers)
    }


def extract_to_cache(
    *,
    manifest: pd.DataFrame,
    raw_root: Path,
    features_dir: Path,
    extractor: FeatureExtractor,
    backbone: str,
    layers: list[str],
    preprocess: PreprocessConfig,
    category: str,
    split: str,
    batch_size: int = 16,
    corruption: str = "none",
) -> CacheResult:
    rows = category_rows(manifest, category, split)
    image_ids = tuple(rows["image_id"])
    directories = shard_directories(
        features_dir=features_dir,
        backbone=backbone,
        layers=layers,
        preprocess=preprocess,
        category=category,
        split=split,
        image_ids=image_ids,
        corruption=corruption,
    )
    missing = [layer for layer, d in directories.items() if not Shard.exists(d)]
    if not missing:
        return CacheResult(directories, ())

    n = len(rows)
    patches: dict[str, torch.Tensor] = {}
    global_token: torch.Tensor | None = None
    grid: tuple[int, int] | None = None
    pos = 0
    source = ManifestSource(rows, raw_root)
    for x in _batches(source, Preprocessor(preprocess), batch_size):
        out = extractor.extract(x, missing)
        b = x.shape[0]
        if grid is None:
            grid = out.grid
            patches = {
                layer: torch.empty(
                    n, *out.patches[layer].shape[1:], dtype=torch.float16
                )
                for layer in missing
            }
            global_token = torch.empty(
                n, out.global_token.shape[-1], dtype=torch.float16
            )
        assert grid is not None and global_token is not None
        for layer in missing:
            patches[layer][pos : pos + b] = out.patches[layer].cpu()
        global_token[pos : pos + b] = out.global_token.cpu()
        pos += b
    assert grid is not None and global_token is not None

    for layer in missing:
        index = {
            "schema_version": SCHEMA_VERSION,
            "backbone": backbone,
            "layer": layer,
            "preprocess": asdict(preprocess),
            "preproc_digest": preprocess.digest(),
            "corruption": corruption,
            "category": category,
            "split": split,
            "grid": list(grid),
            "image_ids": list(image_ids),
        }
        _write_shard(directories[layer], patches[layer], global_token, index)
    return CacheResult(directories, tuple(missing))
