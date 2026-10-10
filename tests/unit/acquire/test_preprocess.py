import numpy as np
import pytest
import torch

from cmad.acquire import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Frame,
    PreprocessConfig,
    Preprocessor,
)


@pytest.mark.parametrize("res", [448, 224])
def test_non_square_input_becomes_square(res: int) -> None:
    image = np.random.default_rng(0).integers(
        0, 256, size=(500, 300, 3), dtype=np.uint8
    )
    out = Preprocessor(PreprocessConfig(res=res))(Frame(image=image))

    assert out.tensor.shape == (3, res, res)
    assert out.tensor.dtype == torch.float32
    assert out.roi_mask is None
    assert out.keep


def test_constant_image_normalises_per_channel() -> None:
    rgb = (200, 100, 50)
    image = np.broadcast_to(np.array(rgb, dtype=np.uint8), (64, 96, 3)).copy()
    out = Preprocessor(PreprocessConfig(res=32))(Frame(image=image))

    for c in range(3):
        expected = (rgb[c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert torch.allclose(out.tensor[c], torch.full((32, 32), expected), atol=1e-5)


def test_digest_stable_and_sensitive() -> None:
    base = PreprocessConfig(res=448)
    assert base.digest() == PreprocessConfig(res=448).digest()
    assert base.digest() != PreprocessConfig(res=224).digest()
    assert base.digest() != PreprocessConfig(res=448, interpolation="bicubic").digest()
    assert base.digest() == PreprocessConfig(res=448, roi_mask=False).digest()


def test_resize_then_crop_removes_the_border() -> None:
    # 100x100 green frame with an 8 px red border
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[..., 1] = 255
    image[:8], image[-8:], image[:, :8], image[:, -8:] = (
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
    )

    whole = Preprocessor(PreprocessConfig(res=40))(Frame(image=image)).tensor
    cropped = Preprocessor(PreprocessConfig(res=40, resize=50))(
        Frame(image=image)
    ).tensor

    assert whole.shape == cropped.shape == (3, 40, 40)
    red = (255 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    # resize 50 -> border is 4 px, the 40 px crop removes 5 px per side: no red is left
    assert not (cropped[0] > red - 0.05).any()
    # without the larger resize the border survives
    assert (whole[0] > red - 0.05).any()


def test_digest_unchanged_when_resize_is_none_so_old_caches_stay_valid() -> None:
    assert (
        PreprocessConfig(res=224).digest() == "4de3c0c010a8cbb49bfc9b50376233c55d93ee76"
    )
    assert (
        PreprocessConfig(res=448).digest() == "293e58667136d7951cfe2f6e4fb0efc749f05a65"
    )
    assert (
        PreprocessConfig(res=224, resize=256).digest()
        != PreprocessConfig(res=224).digest()
    )
    assert (
        PreprocessConfig(res=224, resize=224).digest()
        != PreprocessConfig(res=224).digest()
    )
