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
