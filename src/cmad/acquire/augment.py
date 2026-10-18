"""Reference-image augmentation used by AnomalyDINO."""

from __future__ import annotations

import cv2
import numpy as np

# AnomalyDINO's `agnostic` setting rotates every reference image by these angles (8, not 4)
ROTATIONS_OFFICIAL = (0, 45, 90, 135, 180, 225, 270, 315)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the image centre at the original resolution (bilinear, reflected border).

    Done before the resize, on the full-size image, exactly as in the official code, so the
    reflected border pixels are part of what the backbone sees.
    """
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT
    )
