from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cmad.features import DinoV2Extractor, build_extractor

DEPTH, DIM = 6, 8  # mid index = (2 * 6) // 3 = 4


class FakeDino(nn.Module):
    """Mimics the hub model's get_intermediate_layers contract, incl. ascending order and no duplicates."""

    patch_size = 14
    num_register_tokens = 0

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed_dim = DIM
        self.proj = nn.Linear(3 * 14 * 14, DIM)
        self.cls = nn.Parameter(torch.randn(1, 1, DIM))
        self.blocks = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)

    def get_intermediate_layers(
        self, x, n=1, reshape=False, return_class_token=False, norm=True
    ):
        B = x.shape[0]
        p = (
            x.unfold(2, 14, 14)
            .unfold(3, 14, 14)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(B, -1, 3 * 14 * 14)
        )
        tok = torch.cat([self.cls.expand(B, -1, -1), self.proj(p)], dim=1)
        take = range(DEPTH - n, DEPTH) if isinstance(n, int) else n
        out = []
        for i, blk in enumerate(self.blocks):
            tok = tok + torch.tanh(blk(tok))
            if i in take:
                out.append(tok)
        assert len(out) == len(take)
        if norm:
            out = [self.norm(o) for o in out]
        cls, patches = [o[:, 0] for o in out], [o[:, 1:] for o in out]
        return tuple(zip(patches, cls)) if return_class_token else tuple(patches)


@pytest.fixture
def extractor() -> DinoV2Extractor:
    return DinoV2Extractor("dinov2_vits14", model=FakeDino())


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(2, 3, 28, 42)


def test_shapes_dtypes_and_unit_norms(extractor, x) -> None:
    out = extractor.extract(x, ["last", "mid", "multi"])

    assert out.grid == (2, 3)
    assert out.patches["last"].shape == (2, 6, DIM)
    assert out.patches["mid"].shape == (2, 6, DIM)
    assert out.patches["multi"].shape == (2, 6, 2 * DIM)
    assert out.global_token.shape == (2, DIM)
    for t in [*out.patches.values(), out.global_token]:
        assert t.dtype == torch.float16
        assert torch.allclose(
            t.float().norm(dim=-1), torch.ones(t.shape[:-1]), atol=2e-3
        )


def test_layers_map_to_the_right_blocks_in_any_order(extractor, x) -> None:
    fake = extractor.model

    def direct(block: int) -> torch.Tensor:
        return F.normalize(
            fake.get_intermediate_layers(x, n=[block])[0].float(), dim=-1
        ).half()

    a = extractor.extract(x, ["last", "mid"])
    b = extractor.extract(x, ["mid", "last"])
    c = extractor.extract(x, ["mid", "mid", "2"])

    assert torch.allclose(a.patches["mid"].float(), direct(4).float(), atol=1e-3)
    assert torch.allclose(a.patches["last"].float(), direct(5).float(), atol=1e-3)
    assert torch.equal(a.patches["mid"], b.patches["mid"])
    assert torch.equal(a.patches["last"], b.patches["last"])
    assert torch.allclose(c.patches["2"].float(), direct(2).float(), atol=1e-3)
    assert not torch.allclose(a.patches["mid"].float(), a.patches["last"].float())


def test_multi_is_normalised_concat_of_mid_and_last(extractor, x) -> None:
    out = extractor.extract(x, ["mid", "last", "multi"])
    expected = F.normalize(
        torch.cat([out.patches["mid"].float(), out.patches["last"].float()], dim=-1),
        dim=-1,
    )

    assert torch.allclose(out.patches["multi"].float(), expected, atol=2e-3)


def test_global_token_is_final_block_cls_whatever_layers_requested(
    extractor, x
) -> None:
    only_mid = extractor.extract(x, ["mid"])
    only_last = extractor.extract(x, ["last"])

    assert torch.equal(only_mid.global_token, only_last.global_token)


@pytest.mark.parametrize("bad", ["middle", "6", "-1"])
def test_unknown_layer_raises(extractor, x, bad: str) -> None:
    with pytest.raises(ValueError):
        extractor.extract(x, [bad])


def test_registry_builds_dinov2(x) -> None:
    ext = build_extractor("dinov2_vits14", model=FakeDino())
    assert isinstance(ext, DinoV2Extractor)
    with pytest.raises(KeyError):
        build_extractor("nope")


def _hub_has_vits14() -> bool:
    hub = Path(torch.hub.get_dir())
    return (hub / "facebookresearch_dinov2_main").exists() and (
        hub / "checkpoints" / "dinov2_vits14_pretrain.pth"
    ).exists()


@pytest.mark.skipif(
    not _hub_has_vits14(),
    reason="dinov2_vits14 weights not in the local torch hub cache",
)
def test_real_vits14_at_224() -> None:
    ext = DinoV2Extractor("dinov2_vits14")
    out = ext.extract(torch.randn(1, 3, 224, 224), ["last", "mid", "multi"])

    assert out.grid == (16, 16)
    assert out.patches["last"].shape == (1, 256, 384)
    assert out.patches["multi"].shape == (1, 256, 768)
    assert out.global_token.shape == (1, 384)
    assert torch.allclose(
        out.patches["multi"].float().norm(dim=-1), torch.ones(1, 256), atol=2e-3
    )
