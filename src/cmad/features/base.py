from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FeatureBatch:
    # layer -> B x (h*w) x D, fp16; L2-normalised unless the layer name ends in _raw
    patches: dict[str, torch.Tensor]
    grid: tuple[int, int]
    global_token: torch.Tensor  # B x D (CLS for ViT, GAP for CNN)


class FeatureExtractor(nn.Module):
    default_res: int = 224

    def extract(self, x: torch.Tensor, layers: list[str]) -> FeatureBatch:
        raise NotImplementedError


_REGISTRY: dict[str, type[FeatureExtractor]] = {}


def register(*names: str) -> Callable[[type[FeatureExtractor]], type[FeatureExtractor]]:
    def decorator(cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
        for name in names:
            _REGISTRY[name] = cls
        return cls

    return decorator


def default_resolution(name: str) -> int:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name].default_res


def build_extractor(name: str, **kwargs) -> FeatureExtractor:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](name, **kwargs)
