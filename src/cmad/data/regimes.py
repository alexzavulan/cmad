"""Data regimes: which images are the bank, the validation set, the support faults and the test set.

A `Split` is a pure function of (manifest, category, regime, seed), so every method in a seed sees
the same images, and the leakage rules live in one place:

- `val_normal` is held out of training and never fits anything: a seeded fraction of the training
  normals, or the dataset's own validation folder (MVTec AD 2).
- support abnormals come out of the test pool and are removed from it; for datasets with
  acquisition conditions every condition of a support item leaves the test set (grouped by `item_id`).
- a defect type with fewer than k + 5 candidate test items is never sampled.
- leave-one-defect-type-out (`lodo_<type>`) is the few-abnormal split at k = 5 with one defect type
  given no support: the other types get exactly the support they have at k = 5, and the held-out
  type stays whole in the test set, where its images are faults the method never saw labelled.
  A defect type that mixes several kinds of defect in one image (`MIXED_DEFECT_TYPES`) is never
  support there and is never the held-out type: as support it could show the held-out kind to
  the method, and held out it would not be unseen.
- long-tail (`longtail_r<rho>`) spends the same number of labelled faults as the few-abnormal
  split at k = 5, five per defect type on average, but spreads them unevenly: the defect types
  are put in one seeded order per (category, seed), shared by every ratio, and type i of T gets a
  share proportional to rho^(-i/(T-1)), at least one fault, and never so many that fewer than 5 of
  its items are left for the test set. A mixed defect type counts as an ordinary type here.
- the images in `EXCLUDED_TEST_IMAGES` leave the test pool of every study regime; the two
  reproduction protocols keep the official test set (`uses_full_test_set`).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .ingest.common import item_id_of

TEST_SPLITS = (
    "test",
    "test_public",
)  # MVTec AD, MVTec AD 2 (private splits are never ingested)
VALIDATION_SPLIT = "validation"  # MVTec AD 2 ships its own held-out normals
GUARD_MARGIN = 5  # a type needs k + 5 candidate test items before it may supply support
LODO_PREFIX = "lodo_"
LODO_SHOTS = 5  # labelled faults per seen defect type when one type is held out
# Defect types whose images show several of the category's other defect kinds at once (cable,
# pill, wood and zipper have one). They would blur "this defect type was never labelled".
MIXED_DEFECT_TYPES = frozenset({"combined"})
LONGTAIL_PREFIX = "longtail_r"
LONGTAIL_RATIOS = (1, 3, 10)
# the long-tail budget is this many labelled faults per defect type on average (that of the
# few-abnormal split at the same k); the ratio only changes how it is spread over the types
LONGTAIL_SHOTS = 5

# Test images left out of every study split, with the reason. Decided 2026-11-08, before the
# study's negative-bank runs with labelled faults (earlier development runs were not about the
# gain). The criterion is a foreign object or contamination, of a kind the training normals do
# not show, in an image labelled good, which makes the label ambiguous; every category was
# screened with it.
_LOOSE_BRISTLE = (
    "loose bristle (foreign object) on the background of an image labelled good"
)
EXCLUDED_TEST_IMAGES: dict[str, str] = {
    "mvtec_ad/toothbrush/test/good/005.png": _LOOSE_BRISTLE,
    "mvtec_ad/toothbrush/test/good/006.png": _LOOSE_BRISTLE,
    "mvtec_ad/toothbrush/test/good/007.png": _LOOSE_BRISTLE,
}


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    n_normal_shots: int | None = None  # few-normal regime: train on k normals
    # few-abnormal regime: k labelled faults per defect type (long-tail: k on average)
    n_abnormal_per_type: int | None = None
    shot_selection: str = (
        "random"  # "random" | "contiguous_blocks" (AnomalyDINO's protocol)
    )
    val_normal_frac: float = 0.10
    train_condition: str | None = (
        None  # acquisition condition eligible as support (AD 2)
    )

    @property
    def held_out_type(self) -> str | None:
        """The defect type that gets no support, for a leave-one-defect-type-out regime. It lives
        in the regime name and is not a field, so the stored form of every other regime (and with
        it the hash of its splits and runs) is unchanged."""
        if self.name.startswith(LODO_PREFIX):
            return self.name[len(LODO_PREFIX) :]
        return None

    @property
    def longtail_ratio(self) -> int | None:
        """The head-to-tail ratio of a long-tail regime, from its name like `held_out_type`."""
        if m := re.fullmatch(rf"{LONGTAIL_PREFIX}(\d+)", self.name):
            return int(m[1])
        return None

    def __post_init__(self) -> None:
        if self.shot_selection not in ("random", "contiguous_blocks"):
            raise ValueError(f"unknown shot_selection {self.shot_selection!r}")
        if not 0.0 <= self.val_normal_frac < 1.0:
            raise ValueError(
                f"val_normal_frac must be in [0, 1), got {self.val_normal_frac}"
            )


@dataclass(frozen=True)
class Split:
    dataset: str
    category: str
    seed: int
    spec: RegimeSpec
    train_normal: list[str]
    val_normal: list[str]
    support_abnormal: dict[str, list[str]]  # defect_type -> image_ids
    test: list[str]
    skipped_support_types: list[str] = field(
        default_factory=list
    )  # failed the k + 5 guard
    excluded_test: list[str] = field(
        default_factory=list
    )  # left out of the test pool by EXCLUDED_TEST_IMAGES
    # long-tail regimes only: the type order, the counts before the cap and the realised counts,
    # total and ratio (largest over smallest count actually used)
    longtail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        # omitted when empty, so splits without them keep the sha1 they had before
        for key in ("excluded_test", "longtail"):
            if not d[key]:
                del d[key]
        return d

    def sha1(self) -> str:
        payload = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()


def parse_regime(name: str, **overrides) -> RegimeSpec:
    """oneclass | oneclass_full | few_norm_k<k> | few_abn_k<k> | lodo_<defect type> |
    longtail_r<rho>; keyword overrides set other fields.

    `oneclass_full` is the paper protocol (every training image is the bank, no validation set), the
    one the PatchCore reproduction used; `oneclass` holds 10 % out as `val_normal`.
    """
    if name == "oneclass":
        spec = RegimeSpec(name)
    elif name == "oneclass_full":
        spec = RegimeSpec(name, val_normal_frac=0.0)
    elif m := re.fullmatch(r"few_norm_k(\d+)", name):
        spec = RegimeSpec(name, n_normal_shots=int(m[1]))
    elif m := re.fullmatch(r"few_abn_k(\d+)", name):
        spec = RegimeSpec(name, n_abnormal_per_type=int(m[1]))
    elif name.startswith(LODO_PREFIX) and len(name) > len(LODO_PREFIX):
        spec = RegimeSpec(name, n_abnormal_per_type=LODO_SHOTS)
    elif m := re.fullmatch(rf"{LONGTAIL_PREFIX}(\d+)", name):
        if int(m[1]) < 1:
            raise ValueError(f"{name}: the long-tail ratio must be at least 1")
        spec = RegimeSpec(name, n_abnormal_per_type=LONGTAIL_SHOTS)
    else:
        raise ValueError(
            f"unknown regime {name!r}; expected oneclass, oneclass_full, few_norm_k<k>, "
            "few_abn_k<k>, lodo_<defect type> or longtail_r<rho>"
        )
    clean = {k: v for k, v in overrides.items() if v is not None}
    if clean.get("shot_selection") == "contiguous_blocks":
        # the official blocks come from the full sorted list, so no validation set may overlap them
        clean.setdefault("val_normal_frac", 0.0)
    return RegimeSpec(**{**asdict(spec), **clean})


def uses_full_test_set(spec: RegimeSpec) -> bool:
    """True for the two reproduction protocols, which are compared with published numbers and so
    score the official test set: `oneclass_full` (PatchCore) and contiguous blocks (AnomalyDINO).
    Every other regime leaves `EXCLUDED_TEST_IMAGES` out of the test pool."""
    return spec.name == "oneclass_full" or spec.shot_selection == "contiguous_blocks"


def _rng(seed: int, category: str, purpose: str) -> random.Random:
    # string seeding is hashed with sha512, so it is stable across runs (unlike hash())
    return random.Random(f"{seed}/{category}/{purpose}")


def _shuffled(
    items: Sequence[str], seed: int, category: str, purpose: str
) -> list[str]:
    """One fixed random order per (seed, category, purpose); a k-draw is its first k elements.

    Not `sample(items, k)` per k: CPython's sampler switches algorithm with k and n, so the draws
    for different k are not always nested. A prefix guarantees few_norm_k1 within k2 within ...,
    which is what makes a k-ablation a paired comparison.
    """
    return _rng(seed, category, purpose).sample(list(items), len(items))


def build_split(
    manifest: pd.DataFrame,
    category: str,
    spec: RegimeSpec,
    seed: int,
    excluded: Mapping[str, str] | None = None,
) -> Split:
    """`excluded` defaults to `EXCLUDED_TEST_IMAGES`; it is ignored when `uses_full_test_set`."""
    rows = manifest[manifest["category"] == category]
    if rows.empty:
        raise ValueError(f"no manifest rows for category {category!r}")
    if "item_id" not in rows.columns:  # a manifest ingested before item_id existed
        rows = rows.assign(
            item_id=[
                item_id_of(i, c) for i, c in zip(rows["image_id"], rows["condition"])
            ]
        )
    dataset = str(rows["dataset"].iloc[0])

    train_pool = rows[(rows["orig_split"] == "train") & (rows["label"] == 0)]
    if spec.train_condition is not None:
        train_pool = train_pool[train_pool["condition"] == spec.train_condition]
    train_ids = sorted(train_pool["image_id"])
    if not train_ids:
        raise ValueError(f"no training normals for category {category!r}")

    # validation set: the dataset's own folder if it has one, else a seeded share of the training
    # normals, drawn before (and independently of) any few-shot choice so it is the same for every k
    own_val = rows[(rows["orig_split"] == VALIDATION_SPLIT) & (rows["label"] == 0)]
    if len(own_val):
        val_ids = sorted(own_val["image_id"])
    else:
        n_val = round(spec.val_normal_frac * len(train_ids))
        val_ids = (
            sorted(_rng(seed, category, "val").sample(train_ids, n_val))
            if n_val
            else []
        )
    val_set = set(val_ids)
    fit_ids = [i for i in train_ids if i not in val_set]

    if spec.n_normal_shots is None:
        train_normal = fit_ids
    elif spec.shot_selection == "contiguous_blocks":
        k, lo = spec.n_normal_shots, seed * spec.n_normal_shots
        train_normal = fit_ids[lo : lo + k]
        if len(train_normal) < k:
            raise ValueError(
                f"{category}: seed {seed} needs images {lo}..{lo + k} but only {len(fit_ids)} exist"
            )
    else:
        k = spec.n_normal_shots
        if k > len(fit_ids):
            raise ValueError(
                f"{category}: {k} shots requested but only {len(fit_ids)} normals"
            )
        train_normal = sorted(_shuffled(fit_ids, seed, category, "shots")[:k])

    test_pool = rows[rows["orig_split"].isin(TEST_SPLITS)]
    excluded_test: list[str] = []
    if not uses_full_test_set(spec):
        excluded = EXCLUDED_TEST_IMAGES if excluded is None else excluded
        # dropped before the support draw, so an excluded image is never support either
        drop = test_pool["image_id"].isin(set(excluded))
        excluded_test = sorted(test_pool.loc[drop, "image_id"])
        test_pool = test_pool[~drop]
    support, skipped, removed, longtail = _draw_support(test_pool, spec, seed, category)
    test = sorted(i for i in test_pool["image_id"] if i not in removed)

    split = Split(
        dataset=dataset,
        category=category,
        seed=seed,
        spec=spec,
        train_normal=train_normal,
        val_normal=val_ids,
        support_abnormal=support,
        test=test,
        skipped_support_types=skipped,
        excluded_test=excluded_test,
        longtail=longtail,
    )
    check_disjoint(split, dict(zip(rows["image_id"], rows["item_id"])))
    return split


def longtail_counts(
    order: Sequence[str], sizes: Mapping[str, int], ratio: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Labelled faults per defect type for a long-tail split: (before the cap, after it).

    `order` runs from head to tail and `sizes` counts each type's candidate items. With T types the
    budget is LONGTAIL_SHOTS * T; type i gets round(budget * w_i / sum(w)) with
    w_i = ratio^(-i/(T-1)), rounded half up, at least 1, then capped at its size minus
    GUARD_MARGIN so that every type keeps test images. Budget lost to the cap is not spent
    elsewhere. A type whose cap is below 1 gets 0."""
    t = len(order)
    if t < 2:
        raise ValueError(f"a long-tail split needs two defect types or more, got {t}")
    weights = [ratio ** (-i / (t - 1)) for i in range(t)]
    budget = LONGTAIL_SHOTS * t
    nominal = {
        d: max(1, math.floor(budget * w / sum(weights) + 0.5))
        for d, w in zip(order, weights, strict=True)
    }
    capped = {d: max(0, min(n, sizes[d] - GUARD_MARGIN)) for d, n in nominal.items()}
    return nominal, capped


def _draw_support(
    test_pool: pd.DataFrame, spec: RegimeSpec, seed: int, category: str
) -> tuple[dict[str, list[str]], list[str], set[str], dict]:
    """Support faults per defect type; returns (support, types that get none because they are
    too small, ids to drop from test, the long-tail plan or {}).

    Few-abnormal: k per type, and a type with fewer than k + GUARD_MARGIN items gets none. A
    held-out defect type is passed over: no support, all of its images stay in the test set. The
    draw of every other type does not depend on it, so their support is that of the same k without
    a held-out type. With a held-out type, the mixed defect types are passed over as well.
    Long-tail: the counts of `longtail_counts`, in one seeded type order per (category, seed).

    The n faults of a type are always the first n of one shuffle of its items per (category,
    seed, type), so support is nested across k and the long-tail ratios, and a type given 5 faults
    in a long-tail split has the same five as in the few-abnormal split at k = 5."""
    k = spec.n_abnormal_per_type
    if k is None:
        return {}, [], set(), {}
    anomalous = test_pool[test_pool["label"] == 1]
    held_out = spec.held_out_type
    if held_out is not None and held_out not in set(anomalous["defect_type"]):
        raise ValueError(
            f"{category}: no defect type {held_out!r} to hold out; "
            f"it has {sorted(anomalous['defect_type'].unique())}"
        )
    if held_out in MIXED_DEFECT_TYPES:
        raise ValueError(
            f"{category}: {held_out!r} mixes several defect kinds in one image, so holding it "
            "out would not make its defects unseen"
        )
    without_support = set() if held_out is None else {held_out, *MIXED_DEFECT_TYPES}
    eligible: dict[str, pd.DataFrame] = {}
    items: dict[str, list[str]] = {}
    for defect_type in sorted(anomalous["defect_type"].unique()):
        if defect_type in without_support:
            continue
        of_type = anomalous[anomalous["defect_type"] == defect_type]
        if spec.train_condition is not None:
            of_type = of_type[of_type["condition"] == spec.train_condition]
        eligible[defect_type] = of_type
        items[defect_type] = sorted(of_type["item_id"].unique())

    plan: dict = {}
    if spec.longtail_ratio is not None:
        order = _shuffled(list(items), seed, category, "longtail/order")
        sizes = {d: len(ids) for d, ids in items.items()}
        nominal, counts = longtail_counts(order, sizes, spec.longtail_ratio)
        used = [n for n in counts.values() if n > 0]
        plan = {
            "ratio": spec.longtail_ratio,
            "order": order,
            "nominal": nominal,
            "counts": counts,
            "total": sum(used),
            "realised_ratio": max(used) / min(used),
        }
    else:
        counts = {
            d: k if len(ids) >= k + GUARD_MARGIN else 0 for d, ids in items.items()
        }

    support: dict[str, list[str]] = {}
    skipped: list[str] = []
    removed: set[str] = set()
    for defect_type, ids in items.items():
        n = counts[defect_type]
        if n == 0:
            skipped.append(defect_type)  # kept whole for evaluation
            continue
        chosen = sorted(_shuffled(ids, seed, category, f"support/{defect_type}")[:n])
        of_type = eligible[defect_type]
        support[defect_type] = sorted(
            of_type[of_type["item_id"].isin(chosen)]["image_id"]
        )
        # every condition of a support item leaves the test set: a relit twin would leak
        of_type_all = anomalous[anomalous["defect_type"] == defect_type]
        removed |= set(of_type_all[of_type_all["item_id"].isin(chosen)]["image_id"])
    return support, skipped, removed, plan


def split_path(splits_dir: Path, split: Split) -> Path:
    spec = split.spec
    tag = "" if spec.shot_selection == "random" else f"+{spec.shot_selection}"
    return (
        Path(splits_dir)
        / split.dataset
        / split.category
        / f"{spec.name}{tag}_seed{split.seed}.json"
    )


def save_split(split: Split, splits_dir: Path) -> Path:
    path = split_path(splits_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(split.to_json(), indent=1, sort_keys=True))
    tmp.replace(path)
    return path


def load_split(path: Path) -> Split:
    d = json.loads(Path(path).read_text())
    return Split(**{**d, "spec": RegimeSpec(**d["spec"])})


def get_split(
    manifest: pd.DataFrame, category: str, spec: RegimeSpec, seed: int, splits_dir: Path
) -> Split:
    """Build the split, write it under splits_dir, and refuse to silently differ from a saved one."""
    split = build_split(manifest, category, spec, seed)
    path = split_path(splits_dir, split)
    if path.exists():
        saved = load_split(path)
        if saved.sha1() != split.sha1():
            raise RuntimeError(
                f"{path} differs from what the builder produces now (manifest or builder changed); "
                "delete it deliberately if that is intended"
            )
        return saved
    save_split(split, splits_dir)
    return split


def check_disjoint(split: Split, item_of: Mapping[str, str] | None = None) -> None:
    """Raise if train / validation / support / test overlap (the leakage invariant).

    With `item_of` (image_id -> item_id) the check is per physical item: a relit twin of a support
    image is a different image_id of the same item, and must not be in test either.
    """
    groups: dict[str, Sequence[str]] = {
        "train_normal": split.train_normal,
        "val_normal": split.val_normal,
        "support": [i for ids in split.support_abnormal.values() for i in ids],
        "test": split.test,
    }
    seen: dict[str, str] = {}
    for name, ids in groups.items():
        for image_id in ids:
            if item_of is None:
                if image_id in seen:
                    raise ValueError(
                        f"image {image_id} is in both {seen[image_id]} and {name}"
                    )
                seen[image_id] = name
            else:  # several images of one item may share a group, but not span two
                item = item_of[image_id]
                if seen.setdefault(item, name) != name:
                    raise ValueError(f"item {item} is in both {seen[item]} and {name}")


# the regimes the study uses (the shift regimes are not built yet). "lodo" stands for one regime
# per defect type of each category (see `lodo_regimes`), "longtail" for one per ratio of each
# category with two defect types or more (see `longtail_regimes`).
STUDY_REGIMES = (
    "oneclass",
    *(f"few_norm_k{k}" for k in (1, 2, 4, 8, 16)),
    *(f"few_abn_k{k}" for k in (1, 2, 3, 5, 10)),
    "lodo",
    "longtail",
)


def _defect_types(manifest: pd.DataFrame, category: str) -> set[str]:
    rows = manifest[
        (manifest["category"] == category)
        & manifest["orig_split"].isin(TEST_SPLITS)
        & (manifest["label"] == 1)
    ]
    return set(rows["defect_type"])


def lodo_regimes(manifest: pd.DataFrame, category: str) -> list[str]:
    """One leave-one-defect-type-out regime per defect type of the category, mixed types aside.
    A category with a single such type has none: with that type held out, no fault would be
    labelled at all."""
    types = sorted(_defect_types(manifest, category) - MIXED_DEFECT_TYPES)
    return [f"{LODO_PREFIX}{t}" for t in types] if len(types) > 1 else []


def longtail_regimes(manifest: pd.DataFrame, category: str) -> list[str]:
    """One long-tail regime per ratio in LONGTAIL_RATIOS; none for a category with a single
    defect type, which has no tail."""
    if len(_defect_types(manifest, category)) < 2:
        return []
    return [f"{LONGTAIL_PREFIX}{r}" for r in LONGTAIL_RATIOS]


def parse_seeds(text: str) -> list[int]:
    """ "0-4" -> [0, 1, 2, 3, 4]; "0,2,5" -> [0, 2, 5]."""
    if m := re.fullmatch(r"(\d+)-(\d+)", text):
        return list(range(int(m[1]), int(m[2]) + 1))
    return [int(x) for x in text.split(",")]


def build_all(
    manifest: pd.DataFrame,
    categories: Sequence[str],
    regimes: Sequence[str],
    seeds: Sequence[int],
    splits_dir: Path,
    shot_selection: str | None = None,
) -> list[dict]:
    """Build and save every (category, regime, seed) split; each is checked for leakage.

    Returns one summary row per split. Existing files are kept (and verified against the builder).
    """
    item_of = manifest.set_index("image_id")["item_id"].to_dict()
    rows = []
    for regime in regimes:
        for category in categories:
            expand = {"lodo": lodo_regimes, "longtail": longtail_regimes}.get(regime)
            names = expand(manifest, category) if expand else [regime]
            for name, seed in itertools.product(names, seeds):
                spec = parse_regime(name, shot_selection=shot_selection)
                split = get_split(manifest, category, spec, seed, splits_dir)
                check_disjoint(split, item_of)
                rows.append(
                    {
                        "regime": regime,
                        "name": name,
                        "category": category,
                        "seed": seed,
                        "n_train": len(split.train_normal),
                        "n_val": len(split.val_normal),
                        "n_support": sum(map(len, split.support_abnormal.values())),
                        "n_test": len(split.test),
                        "skipped_types": len(split.skipped_support_types),
                    }
                )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    import os

    import yaml

    parser = argparse.ArgumentParser(prog="python -m cmad.data.regimes")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build and save splits")
    build.add_argument("--tier", default="core", help="configs/tier/<tier>.yaml")
    build.add_argument("--seeds", default="0-4")
    build.add_argument("--regimes", default=",".join(STUDY_REGIMES))
    build.add_argument("--shot-selection", default=None, help="e.g. contiguous_blocks")
    build.add_argument("--data-root", default=None, help="default: $CMAD_DATA_ROOT")
    build.add_argument("--out", default=None, help="default: <data root>/splits")
    args = parser.parse_args(argv)

    root = args.data_root or os.environ.get("CMAD_DATA_ROOT")
    if not root:
        raise SystemExit("CMAD_DATA_ROOT is not set; export it or pass --data-root")
    root = Path(root).expanduser()
    tier_file = (
        Path(__file__).resolve().parents[3] / "configs" / "tier" / f"{args.tier}.yaml"
    )
    tier = yaml.safe_load(tier_file.read_text())
    manifest = pd.read_parquet(root / "manifests" / f"{tier['dataset']}.parquet")
    out = Path(args.out).expanduser() if args.out else root / "splits"
    rows = build_all(
        manifest,
        tier["categories"],
        args.regimes.split(","),
        parse_seeds(args.seeds),
        out,
        args.shot_selection,
    )
    table = pd.DataFrame(rows)
    print(
        table.groupby("regime", sort=False)
        .agg(
            splits=("seed", "size"),
            median_test=("n_test", "median"),
            median_support=("n_support", "median"),
            splits_with_skipped_types=("skipped_types", lambda x: int((x > 0).sum())),
        )
        .to_string()
    )
    print(f"{len(table)} splits under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
