from __future__ import annotations

import warnings
import zlib

import torch

from cmad.features import FeatureBatch

from .base import ScoreOutput
from .knn import KNNScorer, nn_distance

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
SOURCES = ("gt_mask", "top_q", "random")
MIN_VAL_IMAGES = 20  # below this a fold AUROC is too coarse to select with


class NegBankScorer(KNNScorer):
    """kNN against the normal bank, minus lambda times the distance to a bank of fault patches.

    The negative bank holds patches of the labelled support faults. Which patches is the `source`:
    `gt_mask` (patches the defect mask touches; needs `support_masks`), `top_q` (the top `top_frac`
    of each support image's patches by distance to the normal bank; image label only) or `random`
    (as many random patches from the same images; the control for `top_q`, not for `gt_mask`).

    With `support_roi` (the zero-shot foreground mask of the support images) `top_q` and `random`
    select among foreground patches only and `top_frac` counts foreground patches: the bank is
    drawn from the patches that scoring considers, for method and control alike. `gt_mask` ignores
    the ROI, because the label is ground truth for where the defect is.

    Patch score = d_normal - lam * d_abnormal. Both distance maps are computed once per image and
    lambda is applied afterwards, so `score` returns the image scores at every lambda of the grid
    (`aux["curve"]`) alongside those at the operating `self.lam`. lam = 0 is plain kNN.

    `select_lambda` sets the operating lambda by leave-one-support-out AUROC (see there).
    """

    def __init__(
        self,
        lambdas: tuple[float, ...] = LAMBDAS,
        source: str = "gt_mask",
        top_frac: float = 0.10,
        **kw,
    ) -> None:
        super().__init__(**kw)
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
        if not 0.0 < top_frac <= 1.0:
            raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
        self.lambdas, self.source, self.top_frac = tuple(lambdas), source, top_frac
        self.lam = 0.0  # operating point until a selection rule sets it
        self.neg: torch.Tensor | None = None
        self.neg_selection: dict[
            str, torch.Tensor
        ] = {}  # type -> N x P bool, for diagnostics
        # which support image each negative patch came from (leave-one-support-out needs to
        # exclude an image's own patches): index into neg_owner_ids, one entry per row of self.neg
        self.neg_owner: torch.Tensor | None = None
        self.neg_owner_ids: list[str] = []
        self.neg_counts: dict[str, int] = {}  # support image -> patches it contributed
        # what leave-one-support-out needs from the support images, one entry per owner
        self._support_x: list[torch.Tensor] = []  # P x D stacked features
        self._support_dn: list[torch.Tensor] = []  # P distances to the normal bank
        self._support_fg: list[
            torch.Tensor | None
        ] = []  # P bool foreground, if there is an ROI

    def _select(
        self,
        type_name: str,
        ids: list[str],
        d_normal: torch.Tensor,
        masks: dict[str, torch.Tensor] | None,
        roi: torch.Tensor | None,
    ) -> torch.Tensor:
        """N x P bool: which patches of this type's support images enter the negative bank."""
        n, p = d_normal.shape
        if self.source == "gt_mask":
            if masks is None or type_name not in masks:
                raise ValueError(
                    f"source gt_mask needs a patch mask for defect type {type_name!r}"
                )
            sel = masks[type_name].bool()
            if sel.shape != (n, p):
                raise ValueError(
                    f"mask for {type_name!r} is {tuple(sel.shape)}, expected {(n, p)}"
                )
            return sel
        valid = torch.ones(n, p, dtype=torch.bool) if roi is None else roi.bool()
        if valid.shape != (n, p):
            raise ValueError(
                f"roi for {type_name!r} is {tuple(valid.shape)}, expected {(n, p)}"
            )
        sel = torch.zeros(n, p, dtype=torch.bool)
        for row, image_id in enumerate(ids):
            n_valid = int(valid[row].sum())
            if n_valid == 0:
                continue  # fit warns about an image that contributes nothing
            count = max(1, round(self.top_frac * n_valid))
            if self.source == "top_q":
                key = d_normal[row]
            else:
                # random: seeded per support image, so an image's draw does not depend on which
                # other images are in the support set (k = 5 stays inside k = 10)
                g = torch.Generator().manual_seed(
                    self.seed * 1_000_003 + zlib.crc32(image_id.encode())
                )
                key = torch.rand(p, generator=g)
            sel[
                row, key.masked_fill(~valid[row], -float("inf")).topk(count).indices
            ] = True
        return sel

    def fit(
        self,
        train: FeatureBatch,
        roi: torch.Tensor | None = None,
        support: dict[str, FeatureBatch] | None = None,
        support_masks: dict[str, torch.Tensor] | None = None,
        support_ids: dict[str, list[str]] | None = None,
        support_roi: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Normal bank as in KNNScorer; then the negative bank from `support` (defect type -> features).

        `support_masks` (defect type -> N x P bool, patch grid) is only read by source `gt_mask`.
        `support_ids` (defect type -> image ids, in the order of the features) names the images in
        `neg_owner_ids` and seeds the `random` draw per image (required there); other sources name them "<type>#<row>".
        `support_roi` (defect type -> N x P bool foreground) restricts `top_q` and `random` to
        foreground patches (see the class docstring); it is not read by `gt_mask`.
        """
        super().fit(train, roi)
        self.neg, self.neg_selection, self.lam = None, {}, 0.0
        self.neg_owner, self.neg_owner_ids, self.neg_counts = None, [], {}
        self._support_x, self._support_dn, self._support_fg = [], [], []
        if not support:
            return
        parts, owners = [], []
        for type_name, fb in sorted(support.items()):
            x = self._stack(fb)
            ids = (support_ids or {}).get(type_name)
            if ids is None:
                if self.source == "random":
                    # the draw is seeded per image id; row-based names would tie it to the support order
                    raise ValueError(
                        f"source random needs support_ids for defect type {type_name!r}"
                    )
                ids = [f"{type_name}#{i}" for i in range(x.shape[0])]
            if len(ids) != x.shape[0]:
                raise ValueError(
                    f"{type_name!r}: {len(ids)} image ids for {x.shape[0]} support images"
                )
            d_normal = self._patch_distance(fb, self.bank, self.k)
            sel = self._select(
                type_name,
                ids,
                d_normal,
                support_masks,
                (support_roi or {}).get(type_name),
            )
            self.neg_selection[type_name] = sel
            type_roi = (support_roi or {}).get(type_name)
            for row, image_id in enumerate(ids):
                self._support_x.append(x[row])
                self._support_dn.append(d_normal[row])
                self._support_fg.append(
                    None if type_roi is None else type_roi[row].bool()
                )
                count = int(sel[row].sum())
                self.neg_counts[image_id] = count
                if count == 0:
                    warnings.warn(
                        f"support image {image_id} contributes no negative patch (source {self.source})",
                        stacklevel=2,
                    )
                owners.append(
                    torch.full((count,), len(self.neg_owner_ids), dtype=torch.long)
                )
                self.neg_owner_ids.append(image_id)
            parts.append(x[sel])
        neg = torch.cat(parts).to(self.device).float()
        if not len(neg):
            raise ValueError(
                f"no negative patches from {len(self.neg_owner_ids)} support images "
                f"(source {self.source}); refusing to fall back to plain kNN silently"
            )
        self.neg = neg.to(self._dtype())
        self.neg_owner = torch.cat(owners)

    def score(
        self, feats: FeatureBatch, roi: torch.Tensor | None = None
    ) -> ScoreOutput:
        if self.neg is None:
            return super().score(feats, roi)
        d_normal = self._patch_distance(feats, self.bank, self.k)
        # k = 1 on purpose: the bank is small and a larger k would reach across defect types
        d_abn = self._patch_distance(feats, self.neg, 1)
        b = d_normal.shape[0]
        mask = None if roi is None else ~roi.reshape(b, -1).bool()

        def image_at(lam: float) -> tuple[torch.Tensor, torch.Tensor]:
            return self._floor_and_aggregate(d_normal - lam * d_abn, mask)

        curve = {lam: image_at(lam)[0] for lam in self.lambdas}
        img, dist = image_at(self.lam)
        aux = {
            "lambda": self.lam,
            "curve": curve,
            "d_normal": d_normal.reshape(b, *feats.grid),
            "d_abnormal": d_abn.reshape(b, *feats.grid),
        }
        return ScoreOutput(img, dist.reshape(b, *feats.grid), aux)

    def _floor_and_aggregate(
        self, dist: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Image scores and the patch map of `dist` (n x P); `mask` marks ROI-masked patches."""
        if mask is not None:
            # masked patches must never be the image maximum. kNN distances are >= 0, so 0
            # was enough there; here normal patches go negative once lam > 0, so a 0 fill would
            # beat them all. Fill with the lowest valid score, capped at 0 (lam = 0 stays as in kNN).
            low = dist.masked_fill(mask, float("inf")).min(dim=1).values.clamp(max=0)
            dist = torch.where(mask, low[:, None], dist)
        return self._aggregate(dist), dist

    def _distance_per_owner(self, queries: torch.Tensor) -> torch.Tensor:
        """M x O: distance of every query patch to the nearest negative patch of each support image
        (inf for an image that contributed nothing). Excluding one support image from the bank is
        then ignoring one column, so leave-one-out costs one pass over the negatives, not O."""
        q = queries.to(self.device, self._dtype())
        out = torch.full((q.shape[0], len(self.neg_owner_ids)), float("inf"))
        for owner in range(len(self.neg_owner_ids)):
            bank = self.neg[self.neg_owner == owner]
            if len(bank):
                out[:, owner] = nn_distance(q, bank, 1, self.metric).float().cpu()
        return out

    def select_lambda(
        self, val: FeatureBatch | None, val_roi: torch.Tensor | None = None
    ) -> dict:
        """Set `self.lam` by leave-one-support-out AUROC and return what the selection did.

        Fold i holds out support image i. The negative bank is the other support images' patches;
        image i is scored as the faulty example and every `val` image as a normal one, and the
        fold's AUROC is the fraction of normals scored below the held-out fault (ties count half),
        at every lambda of the grid. The fold AUROCs are averaged (each comparison stays inside
        one bank: scores from different banks are not pooled) and the best lambda wins; ties go to
        the smaller lambda. No labelled test image is involved. `vs_lambda0` is the mean paired fold
        difference to lambda = 0 at the chosen lambda and its standard error (null when lambda = 0
        was chosen), so a report can say whether the choice is distinguishable from plain kNN.

        Falls back to lambda = 0 (plain kNN) when there is nothing to select with: fewer than two
        support images, or no validation images. `reason` then says why. With k = 1 and several
        defect types, a fold removes the held-out type's only example, so it measures transfer
        between types, not within-type benefit.
        """
        info: dict = {
            "rule": "leave_one_support_out",
            "selected": False,
            "reason": None,
        }
        n_owner = len(self.neg_owner_ids)
        if self.neg is None or n_owner < 2:
            self.lam = 0.0
            return {**info, "reason": "fewer than two support images"}
        if val is None or not val.patches:
            self.lam = 0.0
            return {**info, "reason": "no validation images"}

        x_val = self._stack(val)
        n_val, p, d = x_val.shape
        if n_val < MIN_VAL_IMAGES:
            warnings.warn(
                f"only {n_val} validation images: a fold AUROC moves in steps of 1/{n_val}, "
                "so the selected lambda is a weak choice",
                stacklevel=2,
            )
        dn_val = self._patch_distance(val, self.bank, self.k)  # V x P
        per_owner_val = self._distance_per_owner(x_val.reshape(-1, d))  # V*P x O
        mask_val = None if val_roi is None else ~val_roi.reshape(n_val, p).bool()

        order = sorted(range(len(self.lambdas)), key=lambda j: self.lambdas[j])
        lams = [self.lambdas[j] for j in order]
        has_patches = torch.bincount(self.neg_owner, minlength=n_owner) > 0
        folds = []
        for held_out in range(n_owner):
            others = has_patches.clone()
            others[held_out] = False
            if not others.any():
                continue  # no bank left for this fold
            d_abn_val = per_owner_val[:, others].min(dim=1).values.reshape(n_val, p)
            per_owner_pos = self._distance_per_owner(self._support_x[held_out])  # P x O
            d_abn_pos = per_owner_pos[:, others].min(dim=1).values[None]  # 1 x P
            fg = self._support_fg[held_out]
            mask_pos = None if fg is None else ~fg[None]
            row = []
            for lam in lams:
                val_scores, _ = self._floor_and_aggregate(
                    dn_val - lam * d_abn_val, mask_val
                )
                pos_score, _ = self._floor_and_aggregate(
                    self._support_dn[held_out][None] - lam * d_abn_pos, mask_pos
                )
                below = (val_scores < pos_score).float() + 0.5 * (
                    val_scores == pos_score
                ).float()
                row.append(float(below.mean()))
            folds.append(row)
        if not folds:
            self.lam = 0.0
            return {**info, "reason": "no support image leaves a bank behind"}

        table = torch.tensor(folds)  # F x L
        mean = table.mean(dim=0)
        best = 0
        for j in range(
            len(lams)
        ):  # ascending lambda, strict >: ties go to the smaller lambda
            if mean[j] > mean[best]:
                best = j
        self.lam = float(lams[best])
        vs_zero = None
        if self.lam != 0.0 and 0.0 in lams and len(folds) > 1:
            # paired by fold: does the selected lambda beat plain kNN beyond fold-to-fold noise?
            diff = table[:, best] - table[:, lams.index(0.0)]
            vs_zero = {
                "mean_diff": float(diff.mean()),
                "se": float(diff.std() / len(folds) ** 0.5),
            }
        return {
            **info,
            "selected": True,
            "lambda": self.lam,
            "n_folds": len(folds),
            "n_val": n_val,
            "vs_lambda0": vs_zero,
            "curve": {f"{lam:g}": float(m) for lam, m in zip(lams, mean)},
            "fold_std": {f"{lam:g}": float(v) for lam, v in zip(lams, table.std(dim=0))}
            if len(folds) > 1
            else None,
        }

    def memory_bytes(self) -> int:
        extra = 0 if self.neg is None else self.neg.numel() * self.neg.element_size()
        return super().memory_bytes() + extra


def floor_masked(patch_maps: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    """Put every ROI-masked patch of a whole set of maps at one shared floor.

    `score` floors masked patches per image, which keeps image scores right but leaves a masked
    background patch of one image above valid patches of another; pixel metrics pool all images, so
    the runner calls this on the concatenated test maps (N x h x w, roi any shape with N x h x w
    elements) before computing them. The floor is the lowest valid score, capped at 0 so that
    lam = 0 stays identical to kNN's 0 fill.
    """
    masked = ~roi.reshape(patch_maps.shape).bool()
    if not masked.any():
        return patch_maps
    valid = patch_maps[~masked]
    floor = valid.min().clamp(max=0) if valid.numel() else patch_maps.new_zeros(())
    return patch_maps.masked_fill(masked, float(floor))
