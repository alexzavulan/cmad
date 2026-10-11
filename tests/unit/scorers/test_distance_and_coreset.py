import pytest
import torch
import torch.nn.functional as F

from cmad.scorers import greedy_coreset, nn_distance


def _unit(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("chunk,query_chunk", [(7, 5), (1000, 2048)])
def test_cosine_matches_brute_force(k: int, chunk: int, query_chunk: int) -> None:
    q, bank = _unit(23, 16, 0), _unit(50, 16, 1)
    got = nn_distance(q, bank, k, "cosine", chunk=chunk, query_chunk=query_chunk)

    expected = (1 - (q @ bank.T)).topk(k, dim=1, largest=False).values.mean(dim=1)
    assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("chunk,query_chunk", [(7, 5), (1000, 2048)])
def test_euclidean_matches_cdist_on_unnormalised_vectors(
    k: int, chunk: int, query_chunk: int
) -> None:
    g = torch.Generator().manual_seed(0)
    q = torch.randn(23, 16, generator=g) * 5
    bank = torch.randn(50, 16, generator=g) * 3 + 1
    got = nn_distance(q, bank, k, "euclidean", chunk=chunk, query_chunk=query_chunk)

    expected = torch.cdist(q, bank).topk(k, dim=1, largest=False).values.mean(dim=1)
    assert torch.allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize("k", [1, 3])
def test_sqeuclidean_is_the_mean_of_squared_neighbour_distances(k: int) -> None:
    g = torch.Generator().manual_seed(1)
    q = torch.randn(23, 16, generator=g) * 5
    bank = torch.randn(50, 16, generator=g) * 3 + 1
    got = nn_distance(q, bank, k, "sqeuclidean", chunk=7, query_chunk=5)

    nearest = (torch.cdist(q, bank) ** 2).topk(k, dim=1, largest=False).values
    assert torch.allclose(got, nearest.mean(dim=1), atol=1e-3)


def test_euclidean_and_cosine_agree_on_unit_vectors() -> None:
    q, bank = _unit(40, 32, 2), _unit(200, 32, 3)
    cos = nn_distance(q, bank, 1, "cosine")
    euc = nn_distance(q, bank, 1, "euclidean")

    assert torch.allclose(euc**2, 2 * cos, atol=1e-4)  # ||a-b||^2 = 2 (1 - cos)


def test_euclidean_sees_vector_length_cosine_does_not() -> None:
    bank = torch.tensor([[1.0, 0.0]])
    same_direction_far = torch.tensor([[10.0, 0.0]])

    # cosine works on unit vectors, so normalising first erases the length difference
    unit_q, unit_b = F.normalize(same_direction_far, dim=-1), F.normalize(bank, dim=-1)
    assert nn_distance(unit_q, unit_b, 1, "cosine").item() == pytest.approx(0.0)
    assert nn_distance(
        same_direction_far, bank, 1, "euclidean"
    ).item() == pytest.approx(9.0)


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError):
        nn_distance(_unit(2, 4, 0), _unit(3, 4, 1), metric="manhattan")


def test_coreset_size_uniqueness_and_determinism() -> None:
    x = torch.randn(500, 32, generator=torch.Generator().manual_seed(0))
    a = greedy_coreset(x, 50, proj_dim=16, seed=0)
    b = greedy_coreset(x, 50, proj_dim=16, seed=0)
    c = greedy_coreset(x, 50, proj_dim=16, seed=1)

    assert a.shape == (50,) and a.unique().numel() == 50
    assert a.min() >= 0 and a.max() < 500
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_coreset_returns_everything_when_asked_for_all() -> None:
    x = torch.randn(10, 4)
    assert torch.equal(greedy_coreset(x, 10), torch.arange(10))
    assert torch.equal(greedy_coreset(x, 99), torch.arange(10))


def test_coreset_keeps_a_tiny_far_cluster_that_random_sampling_would_miss() -> None:
    g = torch.Generator().manual_seed(0)
    big = torch.randn(1000, 8, generator=g) * 0.1
    tiny = torch.randn(3, 8, generator=g) * 0.1 + 20.0  # 3 points, far away
    x = torch.cat([big, tiny])

    idx = greedy_coreset(x, 10, proj_dim=8, seed=0)

    assert (idx >= 1000).any()
