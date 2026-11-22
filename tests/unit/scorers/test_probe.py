"""The linear probe: input, losses, convergence, scoring."""

import pytest
import torch
import torch.nn.functional as F

from cmad.features import FeatureBatch
from cmad.scorers.probe import LinearProbe, global_features, probe_input, probe_loss


def _batch(centre: float, n: int, seed: int, dim: int = 6) -> FeatureBatch:
    g = torch.Generator().manual_seed(seed)
    patches = centre + 0.3 * torch.randn(n, 4, dim, generator=g)
    return FeatureBatch(
        patches={"last": patches.half()},
        grid=(2, 2),
        global_token=(centre + 0.3 * torch.randn(n, 3, generator=g)).half(),
    )


def _test_set() -> FeatureBatch:
    normal, fault = _batch(0.0, 5, 3), _batch(1.0, 5, 4)
    return FeatureBatch(
        patches={"last": torch.cat([normal.patches["last"], fault.patches["last"]])},
        grid=(2, 2),
        global_token=torch.zeros(10, 3).half(),
    )


def test_the_input_is_cls_and_mean_patch_for_a_vit_and_mean_patch_for_a_cnn() -> None:
    batch = _batch(0.0, 2, 0)
    assert probe_input("dinov2_vits14") == "cls_mean_patch"
    assert probe_input("wrn50_in1k") == "mean_patch"
    cnn = global_features(batch, "last", "mean_patch")
    vit = global_features(batch, "last", "cls_mean_patch")
    assert cnn.shape == (2, 6) and vit.shape == (2, 9)
    assert torch.allclose(cnn, batch.patches["last"].float().mean(dim=1))
    assert torch.allclose(vit[:, :3], batch.global_token.float())


def test_the_losses_are_summed_wce_weights_faults_and_focal_has_no_class_weight() -> (
    None
):
    logits = torch.tensor([0.5, -1.0, 2.0])
    labels = torch.tensor([0.0, 1.0, 1.0])
    per_image = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    assert probe_loss(logits, labels, "ce", 4.0, 2.0) == pytest.approx(
        float(per_image.sum())
    )
    wce = probe_loss(logits, labels, "wce", fault_weight=4.0, gamma=2.0)
    assert wce == pytest.approx(float((per_image * torch.tensor([1, 4, 4])).sum()))
    focal = probe_loss(logits, labels, "focal", fault_weight=4.0, gamma=2.0)
    p_true = torch.exp(-per_image)
    assert focal == pytest.approx(float(((1 - p_true) ** 2 * per_image).sum()))


@pytest.mark.parametrize("loss", ["ce", "wce", "focal"])
def test_the_converged_probe_ranks_faults_above_normals_on_separable_data(
    loss: str,
) -> None:
    train, faults = _batch(0.0, 40, 1), {"x": _batch(1.0, 5, 2)}
    probe = LinearProbe("last", "mean_patch", loss=loss)
    probe.fit(train, support=faults)
    out = probe.score(_test_set())

    assert probe.info["converged"] and probe.info["grad_max"] <= 1e-6
    assert out.patch_map is None
    assert out.image_score[5:].min() > out.image_score[:5].max()
    assert probe.info["fault_weight"] == (8.0 if loss == "wce" else 1.0)
    assert (probe.info["n_normal"], probe.info["n_fault"]) == (40, 5)


def test_the_fit_is_the_penalised_minimum_and_does_not_move_with_a_tighter_tolerance() -> (
    None
):
    train, faults = _batch(0.0, 40, 1), {"x": _batch(1.0, 5, 2)}
    loose = LinearProbe("last", "mean_patch", loss="wce", tol=1e-6)
    loose.fit(train, support=faults)
    tight = LinearProbe("last", "mean_patch", loss="wce", tol=1e-10)
    tight.fit(train, support=faults)

    assert torch.allclose(loose.weight, tight.weight, atol=1e-5)
    # the penalty bounds the weights: a smaller C gives a smaller norm
    small = LinearProbe("last", "mean_patch", loss="wce", C=0.1)
    small.fit(train, support=faults)
    assert small.info["weight_norm"] < loose.info["weight_norm"]


def test_a_fit_that_reaches_the_cap_is_marked_not_converged() -> None:
    probe = LinearProbe("last", "mean_patch", max_iter=2)
    probe.fit(_batch(0.0, 40, 1), support={"x": _batch(1.0, 5, 2)})
    assert not probe.info["converged"] and probe.info["n_iter"] <= 2


def test_the_probe_needs_faults_and_a_known_loss() -> None:
    with pytest.raises(ValueError, match="support faults"):
        LinearProbe("last", "mean_patch").fit(_batch(0.0, 4, 0), support={})
    with pytest.raises(ValueError, match="unknown probe loss"):
        LinearProbe("last", "mean_patch", loss="mlp")
