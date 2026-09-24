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
    assert base.digest() != PreprocessConfig(res=448, roi_mask=False).digest()
