"""The AnomalyDINO pieces, each checked against a literal transcription of the official code."""

import cv2
import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from cmad.acquire import MVTEC_MASKED_CATEGORIES, OfficialRoiMask, rotate_image


def _official_rotate(image, angle):  # src/utils.py rotate_image, verbatim
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rot_mat,
        image.shape[1::-1],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_DEFAULT,
    )


def _official_mask(img_features, grid_size, threshold=10, kernel_size=3, border=0.2):
    """DINOv2Wrapper.compute_background_mask with masking_type=True, verbatim (fixed PCA seed)."""
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    first_pc = pca.fit_transform(img_features.astype(np.float32))
    mask = first_pc > threshold
    m = mask.reshape(grid_size)[
        int(grid_size[0] * border) : int(grid_size[0] * (1 - border)),
        int(grid_size[1] * border) : int(grid_size[1] * (1 - border)),
    ]
    if m.sum() <= m.size * 0.35:
        mask = -first_pc > threshold
    mask = cv2.dilate(
        mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)
    ).astype(bool)
    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), np.uint8),
    ).astype(bool)
    return mask.squeeze()


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270, 315])
def test_rotate_image_matches_the_official_function(angle: int) -> None:
    img = np.random.default_rng(angle).integers(
        0, 256, size=(40, 56, 3), dtype=np.uint8
    )

    got = rotate_image(img, angle)

    assert got.shape == img.shape and got.dtype == np.uint8
    assert np.array_equal(got, _official_rotate(img, angle))


def test_rotation_by_zero_is_the_identity() -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(33, 47, 3), dtype=np.uint8)

    assert np.array_equal(rotate_image(img, 0), img)


def _tokens(rng, grid, d, fg, scale):
    """Background near the origin, foreground displaced along one direction by `scale`."""
    n = grid[0] * grid[1]
    direction = np.zeros(d)
    direction[0] = 1.0
    x = rng.normal(0, 0.1, size=(n, d))
    x[fg.reshape(-1)] += scale * direction
    return x.astype(np.float32)


@pytest.mark.parametrize("seed", range(6))
def test_official_mask_matches_the_original_code_including_its_raster_order_morphology(
    seed,
) -> None:
    rng = np.random.default_rng(seed)
    grid = (8, 10)
    fg = np.zeros(grid, bool)
    r0, c0 = rng.integers(1, 4), rng.integers(1, 5)
    fg[r0 : r0 + 4, c0 : c0 + 4] = True
    tokens = _tokens(rng, grid, 12, fg, scale=rng.choice([-1, 1]) * 40.0)

    got = OfficialRoiMask().predict(torch.from_numpy(tokens)[None], grid)[0, 0].numpy()

    assert got.shape == grid and got.dtype == bool
    assert np.array_equal(got.reshape(-1), _official_mask(tokens, grid))


def test_official_mask_finds_a_central_object_whichever_sign_pca_returns() -> None:
    grid = (8, 8)
    fg = np.zeros(grid, bool)
    fg[2:6, 2:6] = True
    masks = []
    for sign in (+1, -1):
        tokens = _tokens(np.random.default_rng(0), grid, 8, fg, scale=sign * 40.0)
        masks.append(
            OfficialRoiMask()
            .predict(torch.from_numpy(tokens)[None], grid)[0, 0]
            .numpy()
        )

    assert np.array_equal(masks[0], masks[1])
    assert masks[0][fg].all()  # the object is kept
    assert (
        not masks[0][0, :].any() and not masks[0][-1, :].any()
    )  # the top and bottom rows are background


def test_the_mask_threshold_is_in_raw_token_scale() -> None:
    grid = (8, 8)
    fg = np.zeros(grid, bool)
    fg[2:6, 2:6] = True
    tiny = _tokens(
        np.random.default_rng(0), grid, 8, fg, scale=1.0
    )  # unit-vector scale

    mask = OfficialRoiMask().predict(torch.from_numpy(tiny)[None], grid)

    assert (
        not mask.any()
    )  # nothing exceeds 10, which is why the cached layer must be raw


def test_the_mvtec_masking_table_is_the_five_categories_of_the_paper() -> None:
    assert MVTEC_MASKED_CATEGORIES == {
        "capsule",
        "hazelnut",
        "pill",
        "screw",
        "toothbrush",
    }
