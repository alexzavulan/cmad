"""Data regimes: which images are the bank, the validation set, the support faults and the test set.

A `Split` is a pure function of (manifest, category, regime, seed), so every method in a seed sees
the same images, and the leakage rules (spec 5.3) live in one place:

- `val_normal` is held out of training and never fits anything: a seeded fraction of the training
  normals, or the dataset's own validation folder (MVTec AD 2).
- support abnormals come out of the test pool and are removed from it; for datasets with
  acquisition conditions every condition of a support item leaves the test set (grouped by `item_id`).
- a defect type with fewer than k + 5 candidate test items is never sampled.
"""

from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    n_normal_shots: int | None = None  # R1: train on k normals
    n_abnormal_per_type: int | None = None  # R2: k labelled faults per defect type
    shot_selection: str = (
        "random"  # "random" | "contiguous_blocks" (AnomalyDINO's protocol)
    )
    val_normal_frac: float = 0.10
    train_condition: str | None = (
        None  # acquisition condition eligible as support (AD 2)
    )

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

    def to_json(self) -> dict:
        return asdict(self)

    def sha1(self) -> str:
        payload = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()


def parse_regime(name: str, **overrides) -> RegimeSpec:
    """oneclass | oneclass_full | few_norm_k<k> | few_abn_k<k>; keyword overrides set other fields.

    `oneclass_full` is the paper protocol (every training image is the bank, no validation set), the
    one the Gate 1 reproduction used; `oneclass` follows spec 5.3 and holds 10 % out as `val_normal`.
    """
    if name == "oneclass":
        spec = RegimeSpec(name)
    elif name == "oneclass_full":
        spec = RegimeSpec(name, val_normal_frac=0.0)
    elif m := re.fullmatch(r"few_norm_k(\d+)", name):
        spec = RegimeSpec(name, n_normal_shots=int(m[1]))
    elif m := re.fullmatch(r"few_abn_k(\d+)", name):
        spec = RegimeSpec(name, n_abnormal_per_type=int(m[1]))
    else:
        raise ValueError(
            f"unknown regime {name!r}; expected oneclass, oneclass_full, few_norm_k<k> or few_abn_k<k>"
        )
    clean = {k: v for k, v in overrides.items() if v is not None}
    if clean.get("shot_selection") == "contiguous_blocks":
        # the official blocks come from the full sorted list, so no validation set may overlap them
        clean.setdefault("val_normal_frac", 0.0)
    return RegimeSpec(**{**asdict(spec), **clean})


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
    manifest: pd.DataFrame, category: str, spec: RegimeSpec, seed: int
) -> Split:
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
    support, skipped, removed = _draw_support(test_pool, spec, seed, category)
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
    )
    check_disjoint(split, dict(zip(rows["image_id"], rows["item_id"])))
    return split


def _draw_support(
    test_pool: pd.DataFrame, spec: RegimeSpec, seed: int, category: str
) -> tuple[dict[str, list[str]], list[str], set[str]]:
    """k faults per defect type; returns (support, types that failed the guard, ids to drop from test)."""
    k = spec.n_abnormal_per_type
    if k is None:
        return {}, [], set()
    anomalous = test_pool[test_pool["label"] == 1]
    support: dict[str, list[str]] = {}
    skipped: list[str] = []
    removed: set[str] = set()
    for defect_type in sorted(anomalous["defect_type"].unique()):
        of_type = anomalous[anomalous["defect_type"] == defect_type]
        eligible = of_type
        if spec.train_condition is not None:
            eligible = of_type[of_type["condition"] == spec.train_condition]
        items = sorted(eligible["item_id"].unique())
        if len(items) < k + GUARD_MARGIN:
            skipped.append(defect_type)  # kept whole for evaluation
            continue
        chosen = sorted(_shuffled(items, seed, category, f"support/{defect_type}")[:k])
        support[defect_type] = sorted(
            eligible[eligible["item_id"].isin(chosen)]["image_id"]
        )
        # every condition of a support item leaves the test set: a relit twin would leak
        removed |= set(of_type[of_type["item_id"].isin(chosen)]["image_id"])
    return support, skipped, removed


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
