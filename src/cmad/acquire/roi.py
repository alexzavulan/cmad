from __future__ import annotations

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA


def _border(grid_mask: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            grid_mask[..., 0, :],
            grid_mask[..., -1, :],
            grid_mask[..., :, 0],
            grid_mask[..., :, -1],
        ],
        axis=-1,
    )


class RoiMask:
    """Zero-shot foreground mask from patch features: PCA first component, threshold, morphological close."""

    def __init__(self, close_kernel: int = 3) -> None:
        self.close_kernel = close_kernel
        self.pca: PCA | None = None
        self.threshold: float = 0.0
        self.sign: float = 1.0

    def fit(self, ref_patches: torch.Tensor, grid: tuple[int, int]) -> RoiMask:
        n, hw, d = ref_patches.shape
        flat = ref_patches.reshape(n * hw, d).float().cpu().numpy()
        self.pca = PCA(n_components=1)
        proj = self.pca.fit_transform(flat)[:, 0]
        self.threshold = float(proj.mean())
        self.sign = 1.0
        # PCA's sign is arbitrary; objects rarely fill the border, so the border side is background
        fg = (proj > self.threshold).reshape(n, *grid)
        if _border(fg).mean() > 0.5:
            self.sign = -1.0
        return self

    def predict(self, patches: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        if self.pca is None:
            raise RuntimeError("RoiMask.predict called before fit")
        b, hw, d = patches.shape
        proj = self.pca.transform(patches.reshape(b * hw, d).float().cpu().numpy())[
            :, 0
        ]
        fg = (
            (self.sign * (proj - self.threshold) > 0).reshape(b, *grid).astype(np.uint8)
        )
        kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
        closed = np.stack([cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel) for m in fg])
        return torch.from_numpy(closed.astype(bool)).unsqueeze(1)


# AnomalyDINO's per-category masking table for MVTec AD (its Table 8): the other ten categories
# are scored without a mask because the background does not hurt (or the mask would cut objects).
MVTEC_MASKED_CATEGORIES = frozenset(
    {"capsule", "hazelnut", "pill", "screw", "toothbrush"}
)


class OfficialRoiMask:
    """AnomalyDINO's zero-shot mask, reproduced from its code (`compute_background_mask`).

    Per image (no shared fit): PCA on that image's own raw DINOv2 tokens, first component above an
    absolute threshold of 10 is foreground; if the central 60 % of the grid keeps at most 35 % of its
    patches the sign is flipped; the mask is then dilated and closed with a 3x3 kernel.

    Faithful quirk: the original runs the two cv2 morphology calls on the flattened (P, 1) mask,
    not on the 2-D grid, so the "enlarge slightly / fill holes" acts along raster order (a patch and
    its left/right neighbours, wrapping across rows). That is reproduced here, because the published
    numbers were produced with it. Needs raw (un-normalised) tokens: the threshold is in their scale.
    """

    def __init__(
        self, threshold: float = 10.0, kernel_size: int = 3, border: float = 0.2
    ) -> None:
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.border = border

    def _one(self, tokens: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
        pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
        first_pc = pca.fit_transform(tokens.astype(np.float32))  # P x 1
        mask = first_pc > self.threshold
        gh, gw = grid
        centre = mask.reshape(grid)[
            int(gh * self.border) : int(gh * (1 - self.border)),
            int(gw * self.border) : int(gw * (1 - self.border)),
        ]
        if centre.sum() <= centre.size * 0.35:
            mask = -first_pc > self.threshold
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(
            bool
        )
        return mask.reshape(grid)

    def predict(self, patches: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """B x P x D raw tokens -> B x 1 x h x w bool foreground mask."""
        masks = [self._one(t, grid) for t in patches.float().cpu().numpy()]
        return torch.from_numpy(np.stack(masks)).unsqueeze(1)
