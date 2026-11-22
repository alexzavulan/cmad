"""A linear classifier on one global feature per image: the supervised baseline.

The probe is trained on the same images and labels the negative bank uses (the training normals
and the labelled support faults) and is a pure classifier: its input never contains a
nearest-neighbour score, which would let it flag new kinds of faults through the one-class
distance.

Input per image:
- DINOv2: the class token concatenated with the mean of the patch tokens of the same block;
- a CNN: the mean of its patch features, i.e. global average pooling of the same locally
  aggregated mid-level features the patch bank uses.
Every dimension is standardised with the training set's mean and standard deviation.

Training: L2-regularised logistic regression fitted to convergence,

    minimise  1/2 |w|^2 + C * sum_i c_i * loss_i     (the bias is not penalised)

with full-batch L-BFGS until the largest gradient component falls below `tol`. The penalty is
what makes the minimum exist: with hundreds of feature dimensions and a few hundred images the
classes are always linearly separable, and without it the loss falls towards 0 for ever as the
weights grow, so any stopping point would be arbitrary. C = 1 is scikit-learn's default and is
fixed in advance; choosing it would need held-out faults, which the study does not have. The
number of iterations is only a safety cap; a run that reaches it is recorded as not converged.

Losses, each isolating one idea:
- `ce`: plain cross-entropy, c_i = 1;
- `wce`: faults weighted by c_i = (training normals / faults), normals 1. With a full-batch
  objective this equals oversampling the faults by that factor, so it covers both;
- `focal`: focal loss with gamma = 2 and no class weight, which reweights by difficulty only.
`ce` and `wce` are convex; focal is not guaranteed to be, so every fit starts from zero weights
and zero bias, which is deterministic.

The image score is the logit. The sigmoid saturates to exactly 1.0 in floating point for
confident faults, and the ties that creates would distort the AUROC.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cmad.features import FeatureBatch
from cmad.scorers.base import ScoreOutput

LOSSES = ("ce", "wce", "focal")
INPUTS = ("cls_mean_patch", "mean_patch")


def probe_input(backbone: str) -> str:
    """The global feature of a backbone: a ViT has a class token, a CNN does not."""
    return "cls_mean_patch" if backbone.startswith("dinov2") else "mean_patch"


def global_features(feats: FeatureBatch, layer: str, kind: str) -> torch.Tensor:
    """One fp32 vector per image."""
    mean_patch = feats.patches[layer].float().mean(dim=1)
    if kind == "mean_patch":
        return mean_patch
    if kind == "cls_mean_patch":
        return torch.cat([feats.global_token.float(), mean_patch], dim=1)
    raise ValueError(f"unknown probe input {kind!r}; use one of {INPUTS}")


def probe_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: str,
    fault_weight: float,
    gamma: float,
) -> torch.Tensor:
    """Per-image losses, weighted; summed (not averaged), as C multiplies the sum."""
    per_image = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if loss == "ce":
        return per_image.sum()
    if loss == "wce":
        weights = torch.where(labels == 1, fault_weight, 1.0).to(per_image.dtype)
        return (weights * per_image).sum()
    if loss == "focal":
        p_true = torch.exp(-per_image)  # the probability given to the true class
        return ((1.0 - p_true) ** gamma * per_image).sum()
    raise ValueError(f"unknown probe loss {loss!r}; use one of {LOSSES}")


class LinearProbe:
    """Fit on normals and faults, score with the logit. Runs on the CPU in float64: the
    training set is a few hundred vectors, and a tight gradient tolerance needs the precision."""

    def __init__(
        self,
        layer: str,
        input: str,
        loss: str = "wce",
        C: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 10_000,
        gamma: float = 2.0,
        extra: int = 0,
    ) -> None:
        if loss not in LOSSES:
            raise ValueError(f"unknown probe loss {loss!r}; use one of {LOSSES}")
        if input not in INPUTS:
            raise ValueError(f"unknown probe input {input!r}; use one of {INPUTS}")
        self.layer, self.input, self.loss = layer, input, loss
        self.C, self.tol, self.max_iter = float(C), float(tol), int(max_iter)
        self.gamma = float(gamma)
        # number of per-image values appended to the global feature (a side variant only)
        self.extra = int(extra)
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.mean = self.std = None
        self.info: dict = {}

    def _x(
        self, feats: FeatureBatch, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = global_features(feats, self.layer, self.input).double()
        if self.extra:
            if extra is None or extra.shape != (len(x), self.extra):
                raise ValueError(f"expected {self.extra} extra value(s) per image")
            x = torch.cat([x, extra.double()], dim=1)
        return x

    def objective(self, x, y, weight, bias, fault_weight) -> torch.Tensor:
        logits = x @ weight + bias
        data = probe_loss(logits, y, self.loss, fault_weight, self.gamma)
        return 0.5 * weight.square().sum() + self.C * data

    def fit(
        self,
        train: FeatureBatch,
        roi: torch.Tensor | None = None,
        support: dict[str, FeatureBatch] | None = None,
        extra_train: torch.Tensor | None = None,
        extra_support: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not support:
            raise ValueError("the linear probe needs labelled support faults")
        normals = self._x(train, extra_train)
        faults = torch.cat(
            [
                self._x(batch, (extra_support or {}).get(t))
                for t, batch in sorted(support.items())
            ]
        )
        x = torch.cat([normals, faults])
        y = torch.cat([torch.zeros(len(normals)), torch.ones(len(faults))]).double()
        self.mean = x.mean(dim=0)
        # a constant dimension would divide by zero; it carries no information either way
        self.std = x.std(dim=0).clamp_min(1e-6)
        x = (x - self.mean) / self.std
        fault_weight = len(normals) / len(faults)

        weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.LBFGS(
            [weight, bias],
            lr=1.0,
            max_iter=self.max_iter,
            max_eval=2 * self.max_iter,
            tolerance_grad=self.tol,
            tolerance_change=0.0,  # stop on the gradient only
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            value = self.objective(x, y, weight, bias, fault_weight)
            value.backward()
            return value

        optimiser.step(closure)
        value = closure()  # the objective and gradient at the solution
        grad = torch.cat([weight.grad.reshape(-1), bias.grad.reshape(-1)])
        grad_max = float(grad.abs().max())
        state = optimiser.state[optimiser._params[0]]
        self.weight, self.bias = weight.detach(), bias.detach()
        self.info = {
            "input": self.input,
            "input_dim": int(x.shape[1]),
            "n_normal": len(normals),
            "n_fault": len(faults),
            "fault_weight": fault_weight if self.loss == "wce" else 1.0,
            "n_iter": int(state.get("n_iter", 0)),
            "converged": grad_max <= self.tol,
            "grad_max": grad_max,  # largest gradient component at the solution
            "objective": float(value.detach()),
            "weight_norm": float(self.weight.norm()),
        }

    @torch.no_grad()
    def score(
        self,
        feats: FeatureBatch,
        roi: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> ScoreOutput:
        if self.weight is None:
            raise RuntimeError("fit the probe first")
        x = (self._x(feats, extra) - self.mean) / self.std
        logits = (x @ self.weight + self.bias).float()
        return ScoreOutput(image_score=logits, patch_map=None)  # no localisation

    def memory_bytes(self) -> int:
        if self.weight is None:
            return 0
        return (self.weight.numel() + 1) * self.weight.element_size()
