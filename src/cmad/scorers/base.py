from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from cmad.features import FeatureBatch


@dataclass
class ScoreOutput:
    image_score: torch.Tensor  # B
    patch_map: torch.Tensor  # B x h x w
    aux: dict = field(default_factory=dict)


class Scorer(Protocol):
    def fit(
        self,
        train: FeatureBatch,
        roi: torch.Tensor | None = None,
        support: dict[str, FeatureBatch] | None = None,
    ) -> None: ...

    def score(
        self, feats: FeatureBatch, roi: torch.Tensor | None = None
    ) -> ScoreOutput: ...

    def memory_bytes(self) -> int: ...
