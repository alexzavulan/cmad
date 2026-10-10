import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cmad.features import FeatureBatch
from cmad.scorers import KNNScorer

GRID = (4, 4)
P, D = 16, 12


def _batch(
    n: int, anomalous: bool, seed: int, raw: bool = False
) -> tuple[FeatureBatch, torch.Tensor]:
    """Normal patches live near one direction; anomalous images get 2 patches in another."""
    g = torch.Generator().manual_seed(seed)
    normal_dir, anomaly_dir = torch.zeros(D), torch.zeros(D)
    normal_dir[0], anomaly_dir[1] = 1.0, 1.0
    x = normal_dir + 0.05 * torch.randn(n, P, D, generator=g)
    where = torch.zeros(n, P, dtype=torch.bool)
    if anomalous:
        where[:, 5] = where[:, 10] = True
        x[where] = anomaly_dir + 0.05 * torch.randn(int(where.sum()), D, generator=g)
    if raw:
        x = x * (1 + 3 * torch.rand(n, P, 1, generator=g))  # varying vector lengths
    else:
        x = F.normalize(x, dim=-1)
    fb = FeatureBatch(patches={"layer": x}, grid=GRID, global_token=x.mean(1))
    return fb, where


def _auroc(scorer: KNNScorer, raw: bool = False) -> float:
    train, _ = _batch(20, False, 0, raw)
    scorer.fit(train)
    good, _ = _batch(15, False, 1, raw)
    bad, _ = _batch(15, True, 2, raw)
    scores = torch.cat([scorer.score(good).image_score, scorer.score(bad).image_score])
    labels = torch.cat([torch.zeros(15), torch.ones(15)])
    return float(roc_auc_score(labels.numpy(), scores.numpy()))


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
@pytest.mark.parametrize("image_agg", ["max", "top1pct"])
def test_separable_data_gives_auroc_one(metric: str, image_agg: str) -> None:
    assert _auroc(KNNScorer(metric=metric, image_agg=image_agg)) == 1.0


def test_coreset_scorer_still_separates() -> None:
    assert _auroc(KNNScorer(coreset_fraction=0.1, coreset_proj_dim=8)) == 1.0


def test_euclidean_on_raw_vectors_with_varying_length() -> None:
    assert _auroc(KNNScorer(metric="euclidean"), raw=True) == 1.0


def test_patch_map_is_highest_at_the_anomalous_patches() -> None:
    train, _ = _batch(20, False, 0)
    scorer = KNNScorer()
    scorer.fit(train)
    bad, _ = _batch(6, True, 3)

    out = scorer.score(bad)

    assert out.patch_map.shape == (6, *GRID)
    top2 = out.patch_map.reshape(6, P).topk(2, dim=1).indices
    assert torch.equal(top2.sort(dim=1).values, torch.tensor([[5, 10]] * 6))


def test_roi_removes_scores_outside_the_mask_at_scoring_time() -> None:
    train, _ = _batch(20, False, 0)
    scorer = KNNScorer()
    scorer.fit(train)
    bad, _ = _batch(4, True, 3)
    roi = torch.ones(4, P, dtype=torch.bool)
    roi[:, 5] = roi[:, 10] = False  # the object mask excludes the defect

    out = scorer.score(bad, roi=roi)

    assert (out.patch_map.reshape(4, P)[:, [5, 10]] == 0).all()
    assert out.image_score.max() < 0.1  # nothing anomalous is left


def test_roi_at_fit_time_filters_the_bank() -> None:
    train, _ = _batch(20, False, 0)
    roi = torch.zeros(20, P, dtype=torch.bool)
    roi[:, :4] = True
    scorer = KNNScorer()
    scorer.fit(train, roi=roi)

    assert scorer.bank.shape[0] == 20 * 4


def test_k_greater_than_one_averages_neighbours() -> None:
    train, _ = _batch(20, False, 0)
    s1, s5 = KNNScorer(k=1), KNNScorer(k=5)
    s1.fit(train)
    s5.fit(train)
    bad, _ = _batch(3, True, 3)

    assert (s5.score(bad).image_score >= s1.score(bad).image_score - 1e-6).all()


def test_memory_bytes_and_errors() -> None:
    train, _ = _batch(20, False, 0)
    scorer = KNNScorer(coreset_fraction=0.25, coreset_proj_dim=8)
    assert scorer.memory_bytes() == 0
    scorer.fit(train)

    assert scorer.bank.shape == (int(20 * P * 0.25), D)
    assert scorer.memory_bytes() == scorer.bank.numel() * 4
    with pytest.raises(RuntimeError):
        KNNScorer().score(train)
    with pytest.raises(ValueError):
        KNNScorer(image_agg="median")
    with pytest.raises(ValueError):
        KNNScorer(metric="manhattan")  # fails at construction, not after a slow fit
