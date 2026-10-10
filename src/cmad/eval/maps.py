from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur (two 1D passes, reflect padding) of B x 1 x H x W.

    Equal to a 2D 33x33 blur but its scratch memory grows with the kernel width, not its
    square: torchvision's 2D version needs ~18 GB for 83 images of 224 x 224.
    """
    k = (
        2 * int(4.0 * sigma + 0.5) + 1
    )  # same 33-tap kernel for sigma=4 as PatchCore / anomalib
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    p = k // 2
    x = F.pad(x, (p, p, p, p), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k))
    return F.conv2d(x, g.view(1, 1, k, 1))


def upsample_and_blur(
    patch_map: torch.Tensor, size: int = 224, sigma: float = 4.0, chunk: int = 16
) -> torch.Tensor:
    """B x h x w patch scores -> B x size x size: bilinear upsample, then Gaussian blur."""
    out = []
    for i in range(
        0, patch_map.shape[0], chunk
    ):  # bound peak memory for large test sets
        x = F.interpolate(
            patch_map[i : i + chunk, None].float(),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        out.append(_gaussian_blur(x, sigma)[:, 0])
    return torch.cat(out)
