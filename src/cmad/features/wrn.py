from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.resnet import ResNet

from .base import FeatureBatch, FeatureExtractor, register


def _l2(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t.float(), dim=-1)


LAYERS = {"cnn_l23", "cnn_l23_raw"}


@register("wrn50_in1k")
class WideResNet50Extractor(FeatureExtractor):
    """PatchCore-style features: layer2 + layer3, 3x3 neighbourhood averaged, on layer2's grid."""

    def __init__(
        self, name: str, device: str = "cpu", model: ResNet | None = None
    ) -> None:
        super().__init__()
        if model is None:
            # IMAGENET1K_V1 is the checkpoint PatchCore used; torchvision verifies its hash prefix on download
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def extract(self, x: torch.Tensor, layers: list[str]) -> FeatureBatch:
        bad = [layer for layer in layers if layer not in LAYERS]
        if bad:
            raise ValueError(
                f"unknown layer(s) {bad}: wrn50_in1k provides {sorted(LAYERS)}"
            )

        m = self.model
        on_cuda = self.device.type == "cuda"
        ctx = torch.autocast("cuda", dtype=torch.float16) if on_cuda else nullcontext()
        with ctx:
            stem = m.maxpool(m.relu(m.bn1(m.conv1(x.to(self.device)))))
            f2 = m.layer2(m.layer1(stem))
            f3 = m.layer3(f2)
            f4 = m.layer4(f3)

        f2, f3 = (
            F.avg_pool2d(f.float(), kernel_size=3, stride=1, padding=1)
            for f in (f2, f3)
        )
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        grid = (f2.shape[-2], f2.shape[-1])
        feats = (
            torch.cat([f2, f3], dim=1).flatten(2).transpose(1, 2)
        )  # B x (h*w) x 1536

        # *_raw keeps the activation magnitudes (PatchCore measures Euclidean distance on them)
        patches = {
            layer: (feats if layer.endswith("_raw") else _l2(feats)).half()
            for layer in layers
        }
        global_token = _l2(f4.float().mean(dim=(2, 3))).half()
        return FeatureBatch(patches=patches, grid=grid, global_token=global_token)
