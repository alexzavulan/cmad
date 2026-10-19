import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cmad.features import FeatureBatch
from cmad.scorers import KNNScorer, NegBankScorer
from cmad.scorers.negbank import floor_masked

pytestmark = pytest.mark.filterwarnings("ignore:only .* validation images")

GRID = (4, 4)
P, D = 16, 12


def _dir(i: int) -> torch.Tensor:
    v = torch.zeros(D)
    v[i] = 1.0
    return v


def _batch(n: int, seed: int, special: dict[int, int] | None = None) -> FeatureBatch:
    """Patches sit near direction 0; `special` maps patch index -> the direction it points to."""
    g = torch.Generator().manual_seed(seed)
    x = _dir(0) + 0.05 * torch.randn(n, P, D, generator=g)
    for patch, direction in (special or {}).items():
        x[:, patch] = _dir(direction) + 0.05 * torch.randn(n, D, generator=g)
    x = F.normalize(x, dim=-1)
    return FeatureBatch(patches={"layer": x}, grid=GRID, global_token=x.mean(1))


def _fitted(source: str = "top_q", **kw) -> NegBankScorer:
    scorer = NegBankScorer(source=source, top_frac=2 / P, **kw)
    # support faults point to direction 1 at patches 5 and 10
    scorer.fit(
        _batch(20, 0),
        support={"crack": _batch(3, 1, {5: 1, 10: 1})},
        support_masks={"crack": _mask({5, 10}, 3)},
        support_ids={"crack": ["c0", "c1", "c2"]},
    )
    return scorer


def _mask(patches: set[int], n: int) -> torch.Tensor:
    m = torch.zeros(n, P, dtype=torch.bool)
    m[:, sorted(patches)] = True
    return m


def test_without_support_it_is_plain_knn() -> None:
    train, test = _batch(20, 0), _batch(6, 3, {7: 1})
    plain, neg = KNNScorer(), NegBankScorer()
    plain.fit(train)
    neg.fit(train)
    out = neg.score(test)
    assert torch.equal(out.image_score, plain.score(test).image_score)
    assert out.aux == {}


def test_lambda_zero_equals_plain_knn() -> None:
    test = _batch(6, 3, {7: 1})
    plain = KNNScorer()
    plain.fit(_batch(20, 0))
    out = _fitted().score(test)
    assert out.aux["lambda"] == 0.0
    assert torch.allclose(out.image_score, plain.score(test).image_score)
    assert torch.allclose(out.aux["curve"][0.0], plain.score(test).image_score)


def test_top_q_picks_the_anomalous_patches() -> None:
    sel = _fitted("top_q").neg_selection["crack"]
    assert sel.sum(dim=1).tolist() == [2, 2, 2]
    assert sel[:, [5, 10]].all()


def test_gt_mask_uses_the_given_patches_and_needs_them() -> None:
    scorer = _fitted("gt_mask")
    assert torch.equal(scorer.neg_selection["crack"], _mask({5, 10}, 3))
    assert scorer.neg.shape[0] == 6
    with pytest.raises(ValueError, match="needs a patch mask"):
        NegBankScorer(source="gt_mask").fit(
            _batch(20, 0), support={"crack": _batch(3, 1)}
        )


def test_random_count_and_determinism() -> None:
    a, b = _fitted("random", seed=4), _fitted("random", seed=4)
    assert a.neg_selection["crack"].sum(dim=1).tolist() == [2, 2, 2]
    assert torch.equal(a.neg_selection["crack"], b.neg_selection["crack"])
    other = _fitted("random", seed=5)
    assert not torch.equal(a.neg_selection["crack"], other.neg_selection["crack"])


def test_random_draw_does_not_depend_on_other_types() -> None:
    train, crack = _batch(20, 0), _batch(3, 1, {5: 1})
    alone, together = (
        NegBankScorer(source="random", seed=1),
        NegBankScorer(source="random", seed=1),
    )
    alone.fit(
        train, support={"crack": crack}, support_ids={"crack": ["c0", "c1", "c2"]}
    )
    together.fit(
        train,
        support={"crack": crack, "dent": _batch(2, 2, {3: 2})},
        support_ids={"crack": ["c0", "c1", "c2"], "dent": ["d0", "d1"]},
    )
    assert torch.equal(alone.neg_selection["crack"], together.neg_selection["crack"])


def test_negative_distance_is_small_at_faults_large_at_normal() -> None:
    out = _fitted("gt_mask").score(_batch(4, 5, {5: 1}))
    d = out.aux["d_abnormal"].reshape(4, P)
    assert d[:, 5].max() < 0.1
    assert d[:, 0].min() > 0.5


def test_lambda_sweep_removes_a_false_positive_class() -> None:
    # nuisance images (a rare patch pointing to direction 2) are as far from the normal bank as
    # real faults (direction 1), so kNN cannot rank them below faults; the negative bank can
    scorer = _fitted("gt_mask")
    nuisance, fault = _batch(10, 6, {3: 2}), _batch(10, 7, {8: 1})
    labels = [0] * 10 + [1] * 10
    curve = {}
    for lam in scorer.lambdas:
        scorer.lam = lam
        s = torch.cat(
            [scorer.score(nuisance).image_score, scorer.score(fault).image_score]
        )
        curve[lam] = roc_auc_score(labels, s.numpy())
    assert curve[1.0] == 1.0
    assert curve[0.0] < 0.8
    scorer.lam = 0.5
    aux_curve = scorer.score(fault).aux["curve"]
    assert set(aux_curve) == set(scorer.lambdas)  # whole sweep from one pass


def test_roi_masked_patches_sit_below_every_valid_patch_at_every_lambda() -> None:
    scorer = _fitted("gt_mask")
    scorer.lam = 1.0
    test = _batch(2, 8, {5: 1})
    roi = torch.ones(2, P, dtype=torch.bool)
    roi[:, 5] = False
    out = scorer.score(test, roi)
    pm = out.patch_map.reshape(2, P)
    assert (pm[:, 5] <= pm[:, [i for i in range(P) if i != 5]].min(dim=1).values).all()
    assert out.aux["curve"][4.0].max() <= scorer.score(test).aux["curve"][4.0].max()


def test_validation() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        NegBankScorer(source="bogus")
    with pytest.raises(ValueError, match="top_frac"):
        NegBankScorer(top_frac=0.0)


def test_masked_background_never_becomes_the_image_maximum_at_positive_lambda() -> None:
    # normal patches go negative once lam > 0, so a masked patch filled with 0 would win the max
    scorer = _fitted("gt_mask")
    scorer.lam = 1.0
    normal = _batch(3, 9)
    roi = torch.ones(3, P, dtype=torch.bool)
    roi[:, 0] = False
    out = scorer.score(normal, roi)
    valid = (out.aux["d_normal"] - out.aux["d_abnormal"]).reshape(3, P)[:, 1:]
    assert torch.allclose(out.image_score, valid.max(dim=1).values)
    assert (out.image_score < 0).all()
    pm = out.patch_map.reshape(3, P)
    assert (pm[:, 0] <= pm[:, 1:].min(dim=1).values).all()


def test_masked_patches_are_zero_when_lambda_is_zero() -> None:
    # keeps lam = 0 identical to KNNScorer's ROI behaviour (the AnomalyDINO reproduction relies on the 0 fill)
    train, test = _batch(20, 0), _batch(4, 3, {7: 1})
    roi = torch.ones(4, P, dtype=torch.bool)
    roi[:, 2] = False
    plain = KNNScorer(image_agg="top1pct")
    plain.fit(train)
    neg = _fitted("gt_mask")
    neg.image_agg = "top1pct"
    want = plain.score(test, roi)
    got = neg.score(test, roi)
    assert torch.allclose(got.patch_map, want.patch_map)
    assert torch.allclose(got.aux["curve"][0.0], want.image_score)


def test_owner_index_and_counts_are_recorded() -> None:
    scorer = NegBankScorer(source="gt_mask")
    scorer.fit(
        _batch(20, 0),
        support={"a": _batch(2, 1), "b": _batch(1, 2)},
        support_masks={"a": _mask({1, 2}, 2), "b": _mask({3}, 1)},
        support_ids={"a": ["a1", "a2"], "b": ["b1"]},
    )
    assert scorer.neg_owner.tolist() == [0, 0, 1, 1, 2]
    assert scorer.neg_owner_ids == ["a1", "a2", "b1"]
    assert scorer.neg_counts == {"a1": 2, "a2": 2, "b1": 1}


def test_empty_negative_bank_is_an_error() -> None:
    scorer = NegBankScorer(source="gt_mask")
    with (
        pytest.raises(ValueError, match="no negative patches"),
        pytest.warns(UserWarning, match="contributes no negative patch"),
    ):
        scorer.fit(
            _batch(20, 0),
            support={"a": _batch(2, 1)},
            support_masks={"a": _mask(set(), 2)},
        )


def test_support_image_with_no_negative_patch_warns() -> None:
    scorer = NegBankScorer(source="gt_mask")
    masks = {"a": _mask({1}, 2)}
    masks["a"][1] = False
    with pytest.warns(UserWarning, match="a2"):
        scorer.fit(
            _batch(20, 0),
            support={"a": _batch(2, 1)},
            support_masks=masks,
            support_ids={"a": ["a1", "a2"]},
        )


def test_random_patches_of_an_image_do_not_depend_on_the_other_support_images() -> None:
    train = _batch(20, 0)
    small = NegBankScorer(source="random", seed=2, top_frac=0.25)
    big = NegBankScorer(source="random", seed=2, top_frac=0.25)
    feats = _batch(3, 1)

    def take(rows: list[int]) -> FeatureBatch:
        return FeatureBatch(
            patches={"layer": feats.patches["layer"][rows]},
            grid=GRID,
            global_token=feats.global_token[rows],
        )

    small.fit(train, support={"a": take([0, 2])}, support_ids={"a": ["i0", "i2"]})
    big.fit(
        train,
        support={"a": take([0, 1, 2])},
        support_ids={"a": ["i0", "i1", "i2"]},
    )
    assert torch.equal(small.neg_selection["a"][0], big.neg_selection["a"][0])
    assert torch.equal(small.neg_selection["a"][1], big.neg_selection["a"][2])


def test_random_requires_support_ids() -> None:
    with pytest.raises(ValueError, match="needs support_ids"):
        NegBankScorer(source="random").fit(_batch(20, 0), support={"a": _batch(2, 1)})


def test_floor_masked_puts_all_masked_patches_below_every_valid_one() -> None:
    scorer = _fitted("gt_mask")
    scorer.lam = 1.0
    normal = _batch(4, 9)
    roi = torch.ones(4, P, dtype=torch.bool)
    roi[:, 0] = False
    maps = scorer.score(normal, roi).patch_map
    assert (
        maps[:, 0, 0].max() > maps.reshape(4, P)[:, 1:].min()
    )  # per-image floor leaks across images
    fixed = floor_masked(maps, roi)
    assert fixed.reshape(4, P)[:, 0].max() <= fixed.reshape(4, P)[:, 1:].min()
    assert torch.equal(fixed.reshape(4, P)[:, 1:], maps.reshape(4, P)[:, 1:])


def test_floor_masked_is_zero_for_nonnegative_maps_and_a_noop_without_mask() -> None:
    maps = torch.rand(3, 4, 4) + 0.5
    roi = torch.ones(3, 16, dtype=torch.bool)
    assert torch.equal(floor_masked(maps, roi), maps)
    roi[:, 3] = False
    out = floor_masked(maps, roi)
    assert (out.reshape(3, 16)[:, 3] == 0).all()  # kNN's 0 fill


def _fit_with_roi(
    source: str, fg: set[int], top_frac: float = 0.2, **kw
) -> NegBankScorer:
    # patch 3 is background that sits far from the normal bank (as dust or texture can); 5 and 10 are the fault
    scorer = NegBankScorer(source=source, top_frac=top_frac, **kw)
    roi = torch.zeros(3, P, dtype=torch.bool)
    roi[:, sorted(fg)] = True
    scorer.fit(
        _batch(20, 0),
        support={"crack": _batch(3, 1, {3: 2, 5: 1, 10: 1})},
        support_masks={"crack": _mask({5, 10}, 3)},
        support_ids={"crack": ["c0", "c1", "c2"]},
        support_roi={"crack": roi},
    )
    return scorer


@pytest.mark.parametrize("source", ["top_q", "random"])
def test_support_roi_restricts_selection_and_counts_foreground_patches(
    source: str,
) -> None:
    fg = set(range(P)) - {3, 0, 1}  # 13 foreground patches
    sel = _fit_with_roi(source, fg).neg_selection["crack"]
    assert not sel[:, [0, 1, 3]].any()
    assert (
        sel.sum(dim=1).tolist() == [round(0.2 * 13)] * 3
    )  # top_frac of the foreground, not of P


def test_top_q_with_roi_still_finds_the_fault_and_without_roi_takes_the_background() -> (
    None
):
    with_roi = _fit_with_roi("top_q", set(range(P)) - {3}, top_frac=2 / 15)
    assert with_roi.neg_selection["crack"][:, [5, 10]].all()
    # no ROI: patch 3 competes for the top slots
    scorer = NegBankScorer(source="top_q", top_frac=3 / P)
    scorer.fit(_batch(20, 0), support={"crack": _batch(3, 1, {3: 2, 5: 1, 10: 1})})
    assert scorer.neg_selection["crack"][:, 3].all()


def test_random_and_top_q_draw_from_the_same_pool_with_the_same_count() -> None:
    fg = set(range(4, P))
    a, b = _fit_with_roi("top_q", fg), _fit_with_roi("random", fg, seed=1)
    assert (
        a.neg_selection["crack"].sum(1).tolist()
        == b.neg_selection["crack"].sum(1).tolist()
    )
    assert not b.neg_selection["crack"][:, :4].any()


def test_gt_mask_ignores_the_support_roi() -> None:
    sel = _fit_with_roi("gt_mask", set(range(P)) - {5}).neg_selection["crack"]
    assert sel[:, 5].all()  # the label wins where the zero-shot mask cut the defect


def test_fully_masked_support_image_contributes_nothing_and_warns() -> None:
    scorer = NegBankScorer(source="top_q")
    roi = torch.ones(2, P, dtype=torch.bool)
    roi[1] = False
    with pytest.warns(UserWarning, match="a#1"):
        scorer.fit(
            _batch(20, 0),
            support={"a": _batch(2, 1, {5: 1})},
            support_roi={"a": roi},
        )
    assert scorer.neg_counts["a#1"] == 0 and scorer.neg_counts["a#0"] > 0


def test_support_roi_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="roi for"):
        NegBankScorer(source="top_q").fit(
            _batch(20, 0),
            support={"a": _batch(2, 1)},
            support_roi={"a": torch.ones(3, P, dtype=torch.bool)},
        )


# ---- leave-one-support-out selection ---------------------------------------------------------


def _take(fb: FeatureBatch, rows: list[int]) -> FeatureBatch:
    return FeatureBatch(
        patches={k: v[rows] for k, v in fb.patches.items()},
        grid=fb.grid,
        global_token=fb.global_token[rows],
    )


def _selection_setup(n_support: int = 4):
    train = _batch(20, 0)
    support = _batch(n_support, 1, {5: 1, 10: 1})  # fault patches point to direction 1
    val = _batch(
        8, 2, {3: 2}
    )  # normal images with a nuisance patch pointing to direction 2
    ids = [f"s{i}" for i in range(n_support)]
    masks = _mask({5, 10}, n_support)
    return train, support, val, ids, masks


def _fit(train, support, ids, masks, source="gt_mask", roi=None, **kw) -> NegBankScorer:
    scorer = NegBankScorer(source=source, lambdas=(1.0, 0.0, 0.5, 2.0), **kw)
    scorer.fit(
        train,
        support={"crack": support},
        support_masks={"crack": masks},
        support_ids={"crack": ids},
        support_roi=None if roi is None else {"crack": roi},
    )
    return scorer


def _auroc_vs_val(held_out: torch.Tensor, val: torch.Tensor) -> float:
    below = (val < held_out).float() + 0.5 * (val == held_out).float()
    return float(below.mean())


@pytest.mark.parametrize("with_roi", [False, True])
def test_selection_equals_refitting_without_each_support_image(with_roi: bool) -> None:
    train, support, val, ids, masks = _selection_setup()
    n = len(ids)
    roi = torch.ones(n, P, dtype=torch.bool)
    roi[:, [0, 3]] = (
        False  # background patches, including the nuisance patch of the val images
    )
    val_roi = torch.ones(val.patches["layer"].shape[0], P, dtype=torch.bool)
    val_roi[:, [0, 3]] = False
    scorer = _fit(train, support, ids, masks, roi=roi if with_roi else None)

    info = scorer.select_lambda(val, val_roi if with_roi else None)

    # reference: really refit the scorer on the other support images and score through score()
    per_fold = []
    for held_out in range(n):
        rest = [i for i in range(n) if i != held_out]
        ref = _fit(train, _take(support, rest), [ids[i] for i in rest], masks[rest])
        val_curve = ref.score(val, val_roi if with_roi else None).aux["curve"]
        pos_curve = ref.score(
            _take(support, [held_out]), roi[[held_out]] if with_roi else None
        ).aux["curve"]
        per_fold.append(
            {lam: _auroc_vs_val(pos_curve[lam][0], val_curve[lam]) for lam in val_curve}
        )
    for lam in (0.0, 0.5, 1.0, 2.0):
        expected = sum(f[lam] for f in per_fold) / n
        assert info["curve"][f"{lam:g}"] == pytest.approx(expected, abs=1e-6)
    assert info["n_folds"] == n and info["selected"]
    assert scorer.lam == info["lambda"]


def test_selection_picks_a_positive_lambda_when_the_negative_bank_helps() -> None:
    # nuisance patches are as far from the normal bank as the faults, so kNN (lambda = 0) cannot
    # rank a held-out fault above them; the other support faults can
    train, support, val, ids, masks = _selection_setup()
    scorer = _fit(train, support, ids, masks)

    info = scorer.select_lambda(val)

    assert info["curve"]["0"] < 0.8
    assert info["lambda"] > 0.0 and info["curve"][f"{info['lambda']:g}"] == 1.0
    assert info["lambda"] == min(
        float(k) for k, v in info["curve"].items() if v == max(info["curve"].values())
    )  # ties go to the smaller lambda


def test_selection_ties_go_to_the_smaller_lambda() -> None:
    # clean val images: the held-out fault is ranked above them at every lambda
    train, support, _, ids, masks = _selection_setup()
    scorer = _fit(train, support, ids, masks)

    info = scorer.select_lambda(_batch(8, 3))

    assert set(info["curve"].values()) == {1.0}
    assert info["lambda"] == 0.0 and scorer.lam == 0.0


def test_selection_falls_back_to_plain_knn_and_says_why() -> None:
    train, support, val, ids, masks = _selection_setup(n_support=1)
    single = _fit(train, support, ids, masks)
    info = single.select_lambda(val)
    assert (
        not info["selected"]
        and "fewer than two" in info["reason"]
        and single.lam == 0.0
    )

    train, support, val, ids, masks = _selection_setup()
    scorer = _fit(train, support, ids, masks)
    scorer.lam = 1.0
    info = scorer.select_lambda(None)
    assert (
        not info["selected"] and "no validation" in info["reason"] and scorer.lam == 0.0
    )


def test_selection_skips_a_fold_that_leaves_no_bank() -> None:
    # only one support image has negative patches: holding it out leaves nothing to score with
    train, support, val, ids, _ = _selection_setup(n_support=3)
    masks = torch.zeros(3, P, dtype=torch.bool)
    masks[0, [5, 10]] = True
    with pytest.warns(UserWarning, match="contributes no negative patch"):
        scorer = _fit(train, support, ids, masks)

    info = scorer.select_lambda(val)

    assert (
        info["selected"] and info["n_folds"] == 2
    )  # folds 1 and 2 hold out an empty image


def test_selection_does_not_let_a_held_out_fault_match_its_own_patches() -> None:
    # two defect types, one image each. Held out, a fault is far from the other type's patches,
    # so at lambda > 0 it cannot beat the nuisance val images (a tie, AUROC 0.5). Left in the
    # bank it would match itself (distance 0) and look perfect: that would make lambda selection
    # pick the largest lambda for nothing.
    train, val = _batch(20, 0), _batch(8, 2, {3: 2})
    type_a, type_b = _batch(1, 1, {5: 1, 10: 1}), _batch(1, 4, {7: 3})
    support = {"a": type_a, "b": type_b}
    masks = {"a": _mask({5, 10}, 1), "b": _mask({7}, 1)}
    ids = {"a": ["a0"], "b": ["b0"]}
    scorer = NegBankScorer(source="gt_mask", lambdas=(0.0, 1.0))
    scorer.fit(train, support=support, support_masks=masks, support_ids=ids)

    info = scorer.select_lambda(val)

    # reference: refit with the other type only
    expected = []
    for held, other in (("a", "b"), ("b", "a")):
        ref = NegBankScorer(source="gt_mask", lambdas=(0.0, 1.0))
        ref.fit(
            train,
            support={other: support[other]},
            support_masks={other: masks[other]},
            support_ids={other: ids[other]},
        )
        v, h = ref.score(val).aux["curve"], ref.score(support[held]).aux["curve"]
        expected.append(_auroc_vs_val(h[1.0][0], v[1.0]))
    assert info["curve"]["1"] == pytest.approx(sum(expected) / 2, abs=1e-6)
    assert info["curve"]["1"] < 0.75  # not the 1.0 that self-matching would give
    assert info["lambda"] == 0.0


def test_selection_reports_the_paired_gain_over_lambda_zero_and_warns_on_a_small_val_set() -> (
    None
):
    train, support, val, ids, masks = _selection_setup()
    scorer = _fit(train, support, ids, masks)

    with pytest.warns(UserWarning, match="only 8 validation images"):
        info = scorer.select_lambda(val)

    assert info["lambda"] > 0.0
    gain = info["vs_lambda0"]
    assert gain["mean_diff"] == pytest.approx(
        info["curve"][f"{info['lambda']:g}"] - info["curve"]["0"], abs=1e-6
    )
    assert gain["se"] >= 0.0
    # nothing to compare when plain kNN won
    clean = _fit(train, support, ids, masks)
    assert clean.select_lambda(_batch(8, 3))["vs_lambda0"] is None
