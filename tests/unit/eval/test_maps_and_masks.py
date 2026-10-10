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
