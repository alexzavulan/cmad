import torch
import torch.nn.functional as F

from cmad.features import FeatureBatch, FeatureExtractor


class FakeExtractor(FeatureExtractor):
    """Deterministic per-sample features (batch-size invariant); records every extract() call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def extract(self, x: torch.Tensor, layers: list[str]) -> FeatureBatch:
        self.calls.append(list(layers))
        pooled = (
            F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)
        )  # B x 4 x 3
        patches = {
            layer: F.normalize(pooled + len(layer), dim=-1).half() for layer in layers
        }
        return FeatureBatch(
            patches=patches,
            grid=(2, 2),
            global_token=F.normalize(x.mean(dim=(2, 3)), dim=-1).half(),
        )
