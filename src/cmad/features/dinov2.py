from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from .base import FeatureBatch, FeatureExtractor, register


def _l2(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t.float(), dim=-1)


@register("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14")
class DinoV2Extractor(FeatureExtractor):
    patch_size = 14
    default_res = 448

    def __init__(
        self, name: str, device: str = "cpu", model: nn.Module | None = None
    ) -> None:
        super().__init__()
        if model is None:
            model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.depth = len(self.model.blocks)
        self.dim = self.model.embed_dim
        self.mid_index = (2 * self.depth) // 3

    def _blocks(self, layer: str) -> list[int]:
        if layer == "last":
            return [self.depth - 1]
        if layer == "mid":
            return [self.mid_index]
        if layer == "multi":
            return [self.mid_index, self.depth - 1]
        if layer.isdigit() and int(layer) < self.depth:
            return [int(layer)]
        raise ValueError(
            f"unknown layer {layer!r}: use last, mid, multi or a block index in [0, {self.depth})"
        )

    @torch.inference_mode()
    def extract(self, x: torch.Tensor, layers: list[str]) -> FeatureBatch:
        _, _, H, W = x.shape
        wanted = {layer: self._blocks(layer) for layer in layers}
        # the hub model returns blocks in ascending depth and rejects duplicates, so
        # ask for the sorted unique set and map results back to the requested names;
        # the final block is always included because global_token comes from it.
        needed = sorted(
            {b for blocks in wanted.values() for b in blocks} | {self.depth - 1}
        )

        on_cuda = self.device.type == "cuda"
        ctx = torch.autocast("cuda", dtype=torch.float16) if on_cuda else nullcontext()
        with ctx:
            outs = self.model.get_intermediate_layers(
                x.to(self.device), n=needed, return_class_token=True, norm=True
            )
        by_block = dict(zip(needed, outs))

        patches = {}
        for layer, blocks in wanted.items():
            parts = [_l2(by_block[b][0]) for b in blocks]
            # a multi-block concat of unit vectors has norm sqrt(len); renormalise so cosine math stays valid
            patches[layer] = _l2(torch.cat(parts, dim=-1)).half()
        global_token = _l2(by_block[self.depth - 1][1]).half()
        return FeatureBatch(
            patches=patches,
            grid=(H // self.patch_size, W // self.patch_size),
            global_token=global_token,
        )
