import pytest
import torch

from cmad.acquire import RoiMask

GRID = (32, 32)
D = 16
CENTRE = (slice(8, 24), slice(8, 24))


def _patches(n: int, centre_cluster: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    clusters = torch.eye(D)[:2] * 5
    labels = torch.full((n, *GRID), 1 - centre_cluster)
    labels[:, CENTRE[0], CENTRE[1]] = centre_cluster
    x = clusters[labels] + 0.1 * torch.randn(n, *GRID, D, generator=g)
    return x.reshape(n, GRID[0] * GRID[1], D)


def _expected() -> torch.Tensor:
    m = torch.zeros(GRID, dtype=torch.bool)
    m[CENTRE] = True
    return m


@pytest.mark.parametrize("centre_cluster", [0, 1])
def test_recovers_centre_regardless_of_pca_sign(centre_cluster: int) -> None:
    roi = RoiMask().fit(_patches(4, centre_cluster), GRID)
    mask = roi.predict(_patches(2, centre_cluster, seed=1), GRID)

    assert mask.shape == (2, 1, *GRID)
    assert mask.dtype == torch.bool
    assert torch.equal(mask[0, 0], _expected())
    assert torch.equal(mask[1, 0], _expected())


def test_close_fills_single_patch_hole() -> None:
    roi = RoiMask().fit(_patches(4, 0), GRID)
    test = _patches(1, 0, seed=1).reshape(1, *GRID, D)
    test[0, 16, 16] = test[0, 0, 0]  # one background-looking patch inside the object
    mask = roi.predict(test.reshape(1, -1, D), GRID)

    assert mask[0, 0, 16, 16]
    assert torch.equal(mask[0, 0], _expected())


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        RoiMask().predict(_patches(1, 0), GRID)
