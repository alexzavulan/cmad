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


def test_raw_layer_keeps_magnitudes_and_matches_normalised_direction(extractor) -> None:
    x = torch.randn(1, 3, 224, 224)

    out = extractor.extract(x, ["cnn_l23", "cnn_l23_raw"])
    raw = out.patches["cnn_l23_raw"].float()
    unit = out.patches["cnn_l23"].float()

    norms = raw.norm(dim=-1)
    assert norms.std() > 0.05 * norms.mean()  # lengths vary, they are not all 1
    assert torch.allclose(F.normalize(raw, dim=-1), unit, atol=3e-3)
    assert out.global_token.float().norm(dim=-1).allclose(torch.ones(1), atol=2e-3)


def _official_embedding(f2: torch.Tensor, f3: torch.Tensor, dim: int) -> torch.Tensor:
    """patchcore-inspection's _embed + MeanMapper + Aggregator, transcribed literally."""

    def patchify(x: torch.Tensor):
        b, c, h, w = x.shape
        u = F.unfold(x, kernel_size=3, stride=1, padding=1)  # B x C*9 x L
        return u.reshape(b, c, 3, 3, -1).permute(0, 4, 1, 2, 3), (h, w)

    (u2, s2), (u3, s3) = patchify(f2), patchify(f3)
    b = u3.shape[0]
    x = u3.reshape(b, s3[0], s3[1], *u3.shape[2:]).permute(0, -3, -2, -1, 1, 2)
    base = x.shape
    x = x.reshape(-1, *x.shape[-2:])
    x = F.interpolate(
        x.unsqueeze(1), size=s2, mode="bilinear", align_corners=False
    ).squeeze(1)
    x = x.reshape(*base[:-2], *s2).permute(0, -2, -1, 1, 2, 3)
    u3 = x.reshape(b, -1, *x.shape[-3:])

    mapped = [
        F.adaptive_avg_pool1d(
            u.reshape(-1, *u.shape[-3:]).reshape(-1, 1, 9 * u.shape[-3]), dim
        ).squeeze(1)
        for u in (u2, u3)
    ]
    stacked = torch.stack(mapped, dim=1)  # N x 2 x dim
    agg = F.adaptive_avg_pool1d(stacked.reshape(len(stacked), 1, -1), dim)
    return agg.reshape(b, s2[0] * s2[1], dim)


def test_patchcore_reduce_matches_the_official_embedding() -> None:
    from cmad.features.wrn import patchcore_reduce

    g = torch.Generator().manual_seed(0)
    f2 = torch.randn(2, 512, 6, 6, generator=g).relu()
    f3 = torch.randn(2, 1024, 3, 3, generator=g).relu()

    got = patchcore_reduce(f2, f3, dim=1024)

    assert got.shape == (2, 36, 1024)
    assert torch.allclose(got, _official_embedding(f2, f3, 1024), atol=1e-5)


def test_pc_layer_is_1024d_unnormalised_and_coexists_with_the_other_layers(
    extractor,
) -> None:
    out = extractor.extract(
        torch.randn(1, 3, 224, 224), ["cnn_l23_raw", "cnn_l23_pc_raw", "cnn_l23"]
    )

    assert out.patches["cnn_l23_pc_raw"].shape == (1, 784, 1024)
    assert out.patches["cnn_l23_raw"].shape == (1, 784, 1536)
    assert out.patches["cnn_l23_pc_raw"].dtype == torch.float16
    norms = out.patches["cnn_l23_pc_raw"].float().norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)
