from itertools import pairwise

import pandas as pd
import pytest

from cmad.data.regimes import (
    GUARD_MARGIN,
    RegimeSpec,
    Split,
    build_split,
    check_disjoint,
    get_split,
    load_split,
    parse_regime,
    save_split,
)


def _row(
    dataset, category, split, name, label, defect="good", condition=None, item=None
):
    image_id = f"{dataset}/{category}/{split}/{name}"
    return {
        "image_id": image_id,
        "item_id": item or image_id,
        "dataset": dataset,
        "category": category,
        "orig_split": split,
        "label": label,
        "defect_type": defect,
        "condition": condition,
    }


def _mvtec(n_train=40, good=6, defects=None) -> pd.DataFrame:
    defects = {"crack": 12, "hole": 7, "tiny": 3} if defects is None else defects
    rows = [_row("mvtec_ad", "toy", "train", f"{i:03d}.png", 0) for i in range(n_train)]
    rows += [
        _row("mvtec_ad", "toy", "test", f"good_{i:03d}.png", 0) for i in range(good)
    ]
    for defect, n in defects.items():
        rows += [
            _row("mvtec_ad", "toy", "test", f"{defect}_{i:03d}.png", 1, defect)
            for i in range(n)
        ]
    return pd.DataFrame(rows)


def _ad2(n_train=30, n_val=6, n_good_items=4, n_bad_items=12) -> pd.DataFrame:
    conds = ["regular", "overexposed", "shift_1"]
    rows = [
        _row(
            "mvtec_ad2",
            "toy",
            "train",
            f"{i:03d}_regular.png",
            0,
            condition="regular",
            item=f"mvtec_ad2/toy/train/good/{i:03d}",
        )
        for i in range(n_train)
    ]
    rows += [
        _row(
            "mvtec_ad2",
            "toy",
            "validation",
            f"{i:03d}_regular.png",
            0,
            condition="regular",
            item=f"mvtec_ad2/toy/validation/good/{i:03d}",
        )
        for i in range(n_val)
    ]
    for i in range(n_good_items):
        rows += [
            _row(
                "mvtec_ad2",
                "toy",
                "test_public",
                f"g{i:03d}_{c}.png",
                0,
                condition=c,
                item=f"mvtec_ad2/toy/test_public/good/{i:03d}",
            )
            for c in conds
        ]
    for i in range(n_bad_items):
        rows += [
            _row(
                "mvtec_ad2",
                "toy",
                "test_public",
                f"b{i:03d}_{c}.png",
                1,
                "bad",
                condition=c,
                item=f"mvtec_ad2/toy/test_public/bad/{i:03d}",
            )
            for c in conds
        ]
    return pd.DataFrame(rows)


def test_parse_regime_names_and_rejects_unknown() -> None:
    assert parse_regime("oneclass").val_normal_frac == 0.10
    assert parse_regime("oneclass_full").val_normal_frac == 0.0
    assert parse_regime("few_norm_k4").n_normal_shots == 4
    assert parse_regime("few_abn_k10").n_abnormal_per_type == 10
    assert (
        parse_regime("few_norm_k1", shot_selection="contiguous_blocks").val_normal_frac
        == 0.0
    )
    with pytest.raises(ValueError, match="unknown regime"):
        parse_regime("longtail_r10")
    with pytest.raises(ValueError):
        RegimeSpec("x", shot_selection="nope")


def test_oneclass_partitions_the_pools_and_holds_out_ten_percent() -> None:
    m = _mvtec()
    s = build_split(m, "toy", parse_regime("oneclass"), seed=0)

    check_disjoint(s)
    assert len(s.val_normal) == 4 and len(s.train_normal) == 36  # 10 % of 40
    train_ids = set(m[m.orig_split == "train"].image_id)
    assert set(s.train_normal) | set(s.val_normal) == train_ids
    assert set(s.test) == set(m[m.orig_split == "test"].image_id)
    assert s.support_abnormal == {}


def test_oneclass_full_is_the_paper_protocol() -> None:
    s = build_split(_mvtec(), "toy", parse_regime("oneclass_full"), seed=0)

    assert s.val_normal == [] and len(s.train_normal) == 40


def test_splits_are_deterministic_and_seeds_differ() -> None:
    m = _mvtec()
    spec = parse_regime("few_abn_k5")
    a, b = build_split(m, "toy", spec, 0), build_split(m, "toy", spec, 0)
    c = build_split(m, "toy", spec, 1)

    assert a == b and a.sha1() == b.sha1()
    assert a.val_normal != c.val_normal and a.support_abnormal != c.support_abnormal


def test_validation_set_does_not_depend_on_k() -> None:
    m = _mvtec()
    vals = [
        tuple(build_split(m, "toy", parse_regime(f"few_norm_k{k}"), 3).val_normal)
        for k in (1, 4, 8)
    ]

    assert len(set(vals)) == 1


def test_few_norm_random_shots_come_from_the_training_pool_minus_validation() -> None:
    m = _mvtec()
    s = build_split(m, "toy", parse_regime("few_norm_k4"), seed=2)

    check_disjoint(s)
    assert len(s.train_normal) == 4 and s.train_normal == sorted(s.train_normal)
    assert set(s.train_normal) <= set(m[m.orig_split == "train"].image_id)
    assert len(s.test) == len(m[m.orig_split == "test"])
    with pytest.raises(ValueError, match="shots requested"):
        build_split(m, "toy", parse_regime("few_norm_k99"), 0)


def test_contiguous_blocks_are_the_sorted_slice_like_anomalydino() -> None:
    m = _mvtec()
    train_ids = sorted(m[m.orig_split == "train"].image_id)
    for seed in (0, 1, 2):
        s = build_split(
            m,
            "toy",
            parse_regime("few_norm_k4", shot_selection="contiguous_blocks"),
            seed,
        )
        assert s.train_normal == train_ids[seed * 4 : seed * 4 + 4]
        assert s.val_normal == []
    with pytest.raises(ValueError, match="only"):
        build_split(
            m,
            "toy",
            parse_regime("few_norm_k16", shot_selection="contiguous_blocks"),
            3,
        )


def test_few_abn_draws_k_per_type_removes_them_from_test_and_applies_the_guard() -> (
    None
):
    m = _mvtec()  # crack 12, hole 7, tiny 3
    s = build_split(m, "toy", parse_regime("few_abn_k5"), seed=0)

    check_disjoint(s)
    assert s.skipped_support_types == ["hole", "tiny"]  # 7 < 5 + 5 and 3 < 5 + 5
    assert (
        list(s.support_abnormal) == ["crack"] and len(s.support_abnormal["crack"]) == 5
    )
    support = set(s.support_abnormal["crack"])
    original_test = set(m[m.orig_split == "test"].image_id)
    assert support <= original_test and not support & set(s.test)
    assert set(s.test) == original_test - support  # nothing else leaves the test set
    assert (
        len(s.train_normal) == 36
    )  # one-class scorers train on the same normals as R0

    ok = build_split(m, "toy", parse_regime("few_abn_k2"), seed=0)
    assert set(ok.support_abnormal) == {
        "crack",
        "hole",
    } and ok.skipped_support_types == ["tiny"]
    assert GUARD_MARGIN == 5


def test_ad2_uses_its_validation_folder_and_groups_support_by_item() -> None:
    m = _ad2()
    spec = parse_regime("few_abn_k3", train_condition="regular")
    s = build_split(m, "toy", spec, seed=1)

    check_disjoint(s)
    assert set(s.val_normal) == set(
        m[m.orig_split == "validation"].image_id
    )  # not 10 % of train
    assert len(s.train_normal) == 30 and not set(s.train_normal) & set(s.val_normal)
    # the three support images are `regular`, and every condition of their items left the test set
    sup = s.support_abnormal["bad"]
    assert len(sup) == 3 and all(i.endswith("_regular.png") for i in sup)
    items = set(m[m.image_id.isin(sup)].item_id)
    assert len(items) == 3
    assert not set(m[m.item_id.isin(items)].image_id) & set(s.test)
    assert len(s.test) == len(m[m.orig_split == "test_public"]) - 3 * 3


def test_ad2_guard_counts_items_not_images() -> None:
    m = _ad2(n_bad_items=7)  # 21 images but only 7 items: k=3 needs 8

    s = build_split(m, "toy", parse_regime("few_abn_k3", train_condition="regular"), 0)

    assert s.skipped_support_types == ["bad"] and s.support_abnormal == {}
    assert len(s.test) == 4 * 3 + 7 * 3  # untouched


def test_a_manifest_without_item_id_still_works() -> None:
    m = _mvtec().drop(columns="item_id")

    s = build_split(m, "toy", parse_regime("few_abn_k2"), 0)

    assert set(s.support_abnormal) == {"crack", "hole"}


def test_round_trip_and_get_split_detects_drift(tmp_path) -> None:
    m = _mvtec()
    spec = parse_regime("few_abn_k2")
    s = get_split(m, "toy", spec, 0, tmp_path)
    path = tmp_path / "mvtec_ad" / "toy" / "few_abn_k2_seed0.json"

    assert path.exists() and load_split(path) == s
    assert get_split(m, "toy", spec, 0, tmp_path) == s  # second call reads it back
    contiguous = parse_regime("few_norm_k4", shot_selection="contiguous_blocks")
    assert save_split(build_split(m, "toy", contiguous, 1), tmp_path).name == (
        "few_norm_k4+contiguous_blocks_seed1.json"
    )
    changed = m[m.image_id != s.test[0]]  # the manifest lost an image
    with pytest.raises(RuntimeError, match="differs"):
        get_split(changed, "toy", spec, 0, tmp_path)


@pytest.mark.parametrize("n_train", [22, 30, 47, 85])
def test_shots_are_nested_across_k(n_train: int) -> None:
    # CPython's sample() switches algorithm with k and n, so separate draws per k are not always
    # nested; a prefix of one shuffle is. Pools of 22..85 are where the plain sampler failed.
    m = _mvtec(n_train=n_train)
    ks = (1, 2, 4, 8, 16)
    for seed in range(25):
        shots = [
            set(
                build_split(m, "toy", parse_regime(f"few_norm_k{k}"), seed).train_normal
            )
            for k in ks
        ]
        assert all(a <= b for a, b in pairwise(shots)), (n_train, seed)


def test_support_is_nested_across_k() -> None:
    m = _mvtec(defects={"crack": 40, "hole": 30})
    for seed in range(25):
        support = [
            set(
                build_split(
                    m, "toy", parse_regime(f"few_abn_k{k}"), seed
                ).support_abnormal["crack"]
            )
            for k in (1, 2, 3, 5, 10)
        ]
        assert all(a <= b for a, b in pairwise(support)), seed


def test_item_level_check_catches_a_relit_twin_of_a_support_image() -> None:
    spec = RegimeSpec("x")
    leaky = Split(
        dataset="d", category="c", seed=0, spec=spec,
        train_normal=["t1"], val_normal=["v1"],
        support_abnormal={"bad": ["b0_regular"]},
        test=["b0_overexposed", "g0"],  # a different image of the support item
        skipped_support_types=[],
    )  # fmt: skip
    item_of = {
        "t1": "t1",
        "v1": "v1",
        "b0_regular": "b0",
        "b0_overexposed": "b0",
        "g0": "g0",
    }

    check_disjoint(leaky)  # image ids alone cannot see it
    with pytest.raises(ValueError, match="item b0 is in both support and test"):
        check_disjoint(leaky, item_of)


def test_image_level_check_rejects_a_repeated_image() -> None:
    dup = Split("d", "c", 0, RegimeSpec("x"), ["a", "a"], [], {}, ["z"])

    with pytest.raises(ValueError, match="image a is in both"):
        check_disjoint(dup)
