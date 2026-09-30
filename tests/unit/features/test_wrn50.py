from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2

from cmad.features import WideResNet50Extractor, build_extractor


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return wide_resnet50_2(weights=None)  # random init: no network, same architecture


@pytest.fixture(scope="module")
def extractor(model) -> WideResNet50Extractor:
    return WideResNet50Extractor("wrn50_in1k", model=model)


def test_shapes_dtypes_and_unit_norms(extractor) -> None:
    out = extractor.extract(torch.randn(2, 3, 224, 224), ["cnn_l23"])

    assert out.grid == (28, 28)
    assert out.patches["cnn_l23"].shape == (2, 784, 512 + 1024)
    assert out.global_token.shape == (2, 2048)
    for t in (out.patches["cnn_l23"], out.global_token):
        assert t.dtype == torch.float16
        assert torch.allclose(
            t.float().norm(dim=-1), torch.ones(t.shape[:-1]), atol=2e-3
        )


def test_non_square_input_grid_follows_layer2(extractor) -> None:
    out = extractor.extract(torch.randn(1, 3, 160, 224), ["cnn_l23"])

    assert out.grid == (20, 28)
    assert out.patches["cnn_l23"].shape == (1, 20 * 28, 1536)


def test_features_are_pooled_layer2_then_upsampled_pooled_layer3(
    extractor, model
) -> None:
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        f2 = model.layer2(
            model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
        )
        f3 = model.layer3(f2)
    pool = lambda f: F.avg_pool2d(f, 3, stride=1, padding=1)
    up3 = F.interpolate(
        pool(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False
    )
    expected = F.normalize(
        torch.cat([pool(f2), up3], dim=1).flatten(2).transpose(1, 2), dim=-1
    )

    got = extractor.extract(x, ["cnn_l23"]).patches["cnn_l23"].float()

    assert torch.allclose(got, expected, atol=2e-3)
    # the first 512 channels come from layer2, the remaining 1024 from layer3
    assert not torch.allclose(got[..., :512], expected[..., 512:1024], atol=2e-3)


@pytest.mark.parametrize("bad", ["last", "mid", "global"])
def test_unknown_layer_raises(extractor, bad: str) -> None:
    with pytest.raises(ValueError):
        extractor.extract(torch.randn(1, 3, 224, 224), [bad])


def test_registry_builds_wrn50(model) -> None:
    assert isinstance(build_extractor("wrn50_in1k", model=model), WideResNet50Extractor)


def _hub_has_wrn50() -> bool:
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "wide_resnet50_2-95faca4d.pth"
    return ckpt.exists()


@pytest.mark.skipif(
    not _hub_has_wrn50(),
    reason="wide_resnet50_2 V1 weights not in the local torch hub cache",
)
def test_real_pretrained_weights_at_224() -> None:
    out = WideResNet50Extractor("wrn50_in1k").extract(
        torch.randn(1, 3, 224, 224), ["cnn_l23"]
    )

    assert out.patches["cnn_l23"].shape == (1, 784, 1536)
    assert torch.isfinite(out.patches["cnn_l23"].float()).all()
