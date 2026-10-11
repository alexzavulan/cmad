from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.resnet import ResNet

from .base import FeatureBatch, FeatureExtractor, register


def _l2(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t.float(), dim=-1)


LAYERS = {"cnn_l23", "cnn_l23_raw", "cnn_l23_pc_raw"}

PC_DIM = (
    1024  # PatchCore's pretrain/target embedding size (official default 1024 > 1024)
)


def patchcore_reduce(
    f2: torch.Tensor, f3: torch.Tensor, dim: int = PC_DIM
) -> torch.Tensor:
    """The official PatchCore embedding (MeanMapper + Aggregator) from raw layer2/layer3 maps.

    f2: B x 512 x h x w, f3: B x 1024 x h3 x w3. Each layer's 3x3 neighbourhood is unfolded to
    C*3*3 values (channel-major, zero padded) and adaptively average-pooled to `dim`; the two
    layers are then concatenated and adaptively pooled again to `dim`. Returns B x (h*w) x dim.

    layer3 has C*9 = 9216 values = exactly 9 per output bin, so its pooling is the plain 3x3
    average of each channel (which commutes with the bilinear upsampling to layer2's grid).
    layer2 has 4608 values for 1024 bins (4.5 each), so its bins straddle channels and the
    unfolded values are needed.
    """
    b, _, h, w = f2.shape
    u2 = F.unfold(f2, kernel_size=3, stride=1, padding=1).transpose(
        1, 2
    )  # B x hw x 4608
    r2 = F.adaptive_avg_pool1d(u2.reshape(b * h * w, 1, -1), dim).reshape(b, h * w, dim)

    p3 = F.avg_pool2d(
        f3, kernel_size=3, stride=1, padding=1
    )  # zero-pad, count_include_pad
    r3 = F.interpolate(p3, size=(h, w), mode="bilinear", align_corners=False)
    r3 = F.adaptive_avg_pool1d(
        r3.flatten(2).transpose(1, 2).reshape(b * h * w, 1, -1), dim
    ).reshape(b, h * w, dim)

    both = torch.cat([r2, r3], dim=-1)  # B x hw x 2*dim
    return F.adaptive_avg_pool1d(both.reshape(b * h * w, 1, -1), dim).reshape(
        b, h * w, dim
    )


@register("wrn50_in1k")
class WideResNet50Extractor(FeatureExtractor):
    """PatchCore-style features: layer2 + layer3, 3x3 neighbourhood averaged, on layer2's grid.

    cnn_l23 / cnn_l23_raw: the 1536-d concatenation (normalised / not).
    cnn_l23_pc_raw: the official 1024-d adaptive-pooled embedding (see `patchcore_reduce`).
    """

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

        f2, f3 = f2.float(), f3.float()
        grid = (f2.shape[-2], f2.shape[-1])
        patches: dict[str, torch.Tensor] = {}
        if any(layer != "cnn_l23_pc_raw" for layer in layers):
            p2, p3 = (
                F.avg_pool2d(f, kernel_size=3, stride=1, padding=1) for f in (f2, f3)
            )
            p3 = F.interpolate(p3, size=grid, mode="bilinear", align_corners=False)
            feats = (
                torch.cat([p2, p3], dim=1).flatten(2).transpose(1, 2)
            )  # B x (h*w) x 1536
            # *_raw keeps the activation magnitudes (PatchCore measures Euclidean distance on them)
            for layer in layers:
                if layer != "cnn_l23_pc_raw":
                    patches[layer] = (
                        feats if layer.endswith("_raw") else _l2(feats)
                    ).half()
        if "cnn_l23_pc_raw" in layers:
            patches["cnn_l23_pc_raw"] = patchcore_reduce(f2, f3).half()
        global_token = _l2(f4.float().mean(dim=(2, 3))).half()
        return FeatureBatch(patches=patches, grid=grid, global_token=global_token)
