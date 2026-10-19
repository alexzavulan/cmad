from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cmad.acquire import Frame, PreprocessConfig, Preprocessor
from cmad.eval import load_masks, upsample_and_blur


def test_constant_map_stays_constant_and_has_the_requested_size() -> None:
    out = upsample_and_blur(torch.full((2, 4, 4), 3.0), size=64)

    assert out.shape == (2, 64, 64)
    assert torch.allclose(out, torch.full_like(out, 3.0), atol=1e-4)


def test_a_single_hot_patch_becomes_a_smooth_blob_in_the_right_cell() -> None:
    m = torch.zeros(1, 4, 4)
    m[0, 1, 2] = (
        1.0  # row 1, col 2 of a 4x4 grid -> pixels y 16..31, x 32..47 at size 64
    )

    out = upsample_and_blur(m, size=64, sigma=4.0)[0]
    y, x = np.unravel_index(int(out.argmax()), out.shape)

    assert 16 <= y < 32 and 32 <= x < 48
    assert out[y, x] > out[0, 0]
    assert out.max() < 1.0  # blurred below the original peak
    assert out.min() >= 0.0


def test_masks_use_the_same_geometry_as_images(tmp_path: Path) -> None:
    # 100x100 black frame with a 12x12 white square off-centre, and a matching mask
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    msk = np.zeros((100, 100), dtype=np.uint8)
    img[30:42, 60:72] = 255
    msk[30:42, 60:72] = 255
    raw = tmp_path / "raw"
    (raw / "ds").mkdir(parents=True)
    Image.fromarray(msk).save(raw / "ds" / "mask.png")
    rows = pd.DataFrame(
        {"image_id": ["ds/img.png"], "label": [1], "mask_path": ["ds/mask.png"]}
    )

    res, resize = 40, 50
    pre = Preprocessor(PreprocessConfig(res=res, resize=resize))
    tensor = pre(Frame(image=img)).tensor[0]  # channel 0, normalised
    mask = load_masks(rows, raw, res, resize)[0]

    bright = tensor > tensor.min() + 0.5 * (tensor.max() - tensor.min())
    assert mask.any() and bright.any()
    inter = (mask & bright).sum().item()
    union = (mask | bright).sum().item()
    assert inter / union > 0.8  # same place, same shape, up to resampling at the edges


def test_good_images_get_empty_masks_and_missing_masks_on_anomalies_raise(
    tmp_path: Path,
) -> None:
    good = pd.DataFrame({"image_id": ["a"], "label": [0], "mask_path": [None]})
    assert not load_masks(good, tmp_path, 32, 40).any()

    bad = pd.DataFrame({"image_id": ["b"], "label": [1], "mask_path": [None]})
    with pytest.raises(ValueError):
        load_masks(bad, tmp_path, 32, 40)


def test_separable_blur_matches_torchvisions_2d_blur() -> None:
    from torchvision.transforms.v2 import functional as TF

    m = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(0))
    ours = upsample_and_blur(m, size=64, sigma=4.0)
    up = torch.nn.functional.interpolate(
        m[:, None], size=(64, 64), mode="bilinear", align_corners=False
    )
    ref = TF.gaussian_blur(up, kernel_size=[33, 33], sigma=[4.0, 4.0])[:, 0]

    assert torch.allclose(ours, ref, atol=1e-5)


def test_chunking_does_not_change_the_result() -> None:
    m = torch.rand(10, 4, 4, generator=torch.Generator().manual_seed(1))

    assert torch.allclose(
        upsample_and_blur(m, size=32, chunk=3),
        upsample_and_blur(m, size=32, chunk=64),
        atol=1e-6,
    )


def _png_rows(tmp_path: Path, masks: list[np.ndarray]) -> pd.DataFrame:
    (tmp_path / "ds").mkdir(exist_ok=True)
    paths = []
    for i, m in enumerate(masks):
        Image.fromarray((m * 255).astype(np.uint8)).save(tmp_path / "ds" / f"m{i}.png")
        paths.append(f"ds/m{i}.png")
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(len(masks))],
            "label": 1,
            "mask_path": paths,
        }
    )


def test_coverage_is_the_defect_area_fraction_of_each_patch(tmp_path: Path) -> None:
    from cmad.eval import mask_coverage

    m = np.zeros((448, 448), dtype=bool)
    m[0:14, 0:14] = True  # exactly patch 0
    m[28:35, 14:28] = True  # half of patch (row 2, col 1)
    cov = mask_coverage(_png_rows(tmp_path, [m]), tmp_path, 448, (32, 32))[0]
    assert cov[0] == pytest.approx(1.0)
    assert cov[2 * 32 + 1] == pytest.approx(0.5)
    assert cov.sum() == pytest.approx(1.5)


def test_a_single_defect_pixel_is_never_lost_to_the_downscale(tmp_path: Path) -> None:
    # nearest-neighbour keeps every 2nd-3rd pixel of a 1024 px mask, so tiny defects can vanish
    from cmad.eval import support_patch_masks

    g = np.random.default_rng(0)
    masks = []
    for _ in range(40):
        m = np.zeros((1024, 1024), dtype=bool)
        m[g.integers(0, 1024), g.integers(0, 1024)] = True
        masks.append(m)
    rows = _png_rows(tmp_path, masks)
    patches = support_patch_masks(rows, tmp_path, 448, (32, 32))
    assert (patches.sum(dim=1) >= 1).all()
    nearest = load_masks(rows, tmp_path, 448)
    assert (nearest.flatten(1).sum(dim=1) == 0).any()  # the failure this guards against


def test_every_defect_pixel_marks_the_patch_it_lies_in(tmp_path: Path) -> None:
    from cmad.eval import support_patch_masks

    m = np.zeros((1024, 1024), dtype=bool)
    m[300:305, 700:703] = True  # 1024 px / 32 = 32 px cells: row 9, col 21
    out = support_patch_masks(_png_rows(tmp_path, [m]), tmp_path, 448, (32, 32))[0]
    assert out.nonzero().flatten().tolist() == [9 * 32 + 21]


def test_best_covered_patch_is_kept_whatever_the_threshold() -> None:
    from cmad.eval import coverage_to_patches

    cov = torch.zeros(2, 16)
    cov[0, 3], cov[0, 4] = 0.05, 0.02  # both below the threshold: the best one stays
    cov[1, 5], cov[1, 6] = 0.9, 0.2  # 5 passes, 6 is dropped by the threshold
    out = coverage_to_patches(cov, min_coverage=0.5)
    assert out[0].nonzero().flatten().tolist() == [3]
    assert out[1].nonzero().flatten().tolist() == [5]
    assert coverage_to_patches(cov)[1].nonzero().flatten().tolist() == [
        5,
        6,
    ]  # default: any overlap


def test_empty_masks_stay_empty_and_a_grid_that_does_not_divide_the_image_raises(
    tmp_path: Path,
) -> None:
    from cmad.eval import coverage_to_patches, mask_coverage

    rows = _png_rows(tmp_path, [np.zeros((64, 64), dtype=bool)])
    assert not coverage_to_patches(mask_coverage(rows, tmp_path, 64, (8, 8))).any()
    with pytest.raises(ValueError, match="does not divide"):
        mask_coverage(rows, tmp_path, 64, (7, 7))


def test_support_patch_masks_follow_the_image_geometry(tmp_path: Path) -> None:
    from cmad.eval import support_patch_masks

    # 60 x 80 masks: short side 60 -> 50, centre crop 40 x 40 (the crop drops the sides)
    edge = np.zeros((60, 80), dtype=bool)
    edge[10:20, 0:5] = True  # far left: cut away by the crop
    middle = np.zeros((60, 80), dtype=bool)
    middle[25:35, 38:48] = True  # near the centre: survives
    rows = _png_rows(tmp_path, [edge, middle])
    patches = support_patch_masks(rows, tmp_path, res=40, grid=(4, 4), resize=50)
    assert patches.shape == (2, 16)
    assert not patches[0].any()
    assert patches[1].any()
    # the surviving patches are where the geometry-aligned pixel mask puts them
    aligned = load_masks(rows, tmp_path, 40, 50)[1]
    cells = aligned.reshape(4, 10, 4, 10).any(dim=3).any(dim=1).reshape(-1)
    assert torch.equal(patches[1] & cells, cells)  # nothing nearest sees is missing
