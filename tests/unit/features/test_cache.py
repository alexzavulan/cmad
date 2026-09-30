from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from _fakes import FakeExtractor
from PIL import Image

from cmad.acquire import ManifestSource, PreprocessConfig, Preprocessor
from cmad.features import CacheKey, Shard, extract_to_cache

RES = 16
CFG = PreprocessConfig(res=RES)


def _make_dataset(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    raw = tmp_path / "raw"
    rng = np.random.default_rng(0)
    rows = []
    for split, n in [("train", 5), ("test", 3)]:
        for i in range(n):
            rel = f"ds/cat_a/{split}/{i:03d}.png"
            (raw / rel).parent.mkdir(parents=True, exist_ok=True)
            arr = rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
            Image.fromarray(arr).save(raw / rel)
            rows.append(
                {
                    "image_id": rel,
                    "image_path": rel,
                    "category": "cat_a",
                    "orig_split": split,
                    "label": 0,
                    "defect_type": "good",
                }
            )
    # shuffled on purpose: the cache must not depend on manifest row order
    return pd.DataFrame(rows).sample(frac=1, random_state=1), raw


@pytest.fixture
def dataset(tmp_path: Path) -> dict:
    manifest, raw = _make_dataset(tmp_path)
    return {
        "manifest": manifest,
        "raw_root": raw,
        "features_dir": tmp_path / "features",
        "backbone": "fake",
        "preprocess": CFG,
        "category": "cat_a",
    }


def _key(**overrides) -> CacheKey:
    base = CacheKey("fake", "last", 448, "abc", "cat_a", "train", ("a", "b"))
    return replace(base, **overrides)


def test_key_digest_stable_and_sensitive() -> None:
    assert _key().digest() == _key().digest()
    for change in [
        {"backbone": "other"},
        {"layer": "mid"},
        {"res": 224},
        {"preproc_digest": "xyz"},
        {"category": "cat_b"},
        {"split": "test"},
        {"image_ids": ("a", "c")},
        {"corruption": "blur_3"},
    ]:
        assert _key(**change).digest() != _key().digest(), change


def test_extract_writes_shards_then_second_call_is_a_pure_cache_hit(dataset) -> None:
    ex = FakeExtractor()
    first = extract_to_cache(
        **dataset, extractor=ex, layers=["last", "mid"], split="train", batch_size=2
    )

    assert first.extracted == ("last", "mid")
    assert ex.calls and all(call == ["last", "mid"] for call in ex.calls)
    shard = Shard(first.directories["last"])
    assert shard.grid == (2, 2)
    assert shard.patches().shape == (5, 4, 3) and shard.patches().dtype == torch.float16
    assert shard.global_token().shape == (5, 3)
    assert shard.image_ids == sorted(shard.image_ids)
    assert shard.index["split"] == "train" and shard.index["backbone"] == "fake"

    calls_before = len(ex.calls)
    second = extract_to_cache(
        **dataset, extractor=ex, layers=["last", "mid"], split="train", batch_size=2
    )
    assert second.extracted == ()
    assert len(ex.calls) == calls_before
    assert torch.equal(Shard(second.directories["last"]).patches(), shard.patches())


def test_adding_a_layer_extracts_only_that_layer(dataset) -> None:
    ex = FakeExtractor()
    extract_to_cache(**dataset, extractor=ex, layers=["last"], split="train")
    ex.calls.clear()
    result = extract_to_cache(
        **dataset, extractor=ex, layers=["last", "multi"], split="train"
    )

    assert result.extracted == ("multi",)
    assert ex.calls == [["multi"]]


def test_features_match_direct_extraction_for_any_batch_size(dataset) -> None:
    ids = sorted(dataset["manifest"].query("orig_split == 'train'")["image_id"])
    pre = Preprocessor(CFG)
    rows = dataset["manifest"].set_index("image_id").loc[ids].reset_index()
    xs = torch.stack([pre(f).tensor for f in ManifestSource(rows, dataset["raw_root"])])
    expected = FakeExtractor().extract(xs, ["last"]).patches["last"]

    for batch_size in (1, 2, 16):
        feats = dataset["features_dir"] / str(batch_size)
        result = extract_to_cache(
            **{**dataset, "features_dir": feats},
            extractor=FakeExtractor(),
            layers=["last"],
            split="train",
            batch_size=batch_size,
        )
        shard = Shard(result.directories["last"])
        assert shard.image_ids == ids
        assert torch.equal(shard.patches(), expected)


def test_select_returns_rows_in_requested_order(dataset) -> None:
    result = extract_to_cache(
        **dataset, extractor=FakeExtractor(), layers=["last"], split="train"
    )
    shard = Shard(result.directories["last"])
    want = [shard.image_ids[3], shard.image_ids[0]]

    picked = shard.patches(want)

    assert torch.equal(picked, shard.patches()[[3, 0]])
    assert torch.equal(shard.global_token(want), shard.global_token()[[3, 0]])


def test_shard_without_index_is_treated_as_missing(dataset) -> None:
    ex = FakeExtractor()
    first = extract_to_cache(**dataset, extractor=ex, layers=["last"], split="train")
    directory = first.directories["last"]
    (directory / "index.json").unlink()
    (directory / "index.json.tmp").write_text("{}")  # leftover from a killed job

    assert not Shard.exists(directory)
    again = extract_to_cache(**dataset, extractor=ex, layers=["last"], split="train")
    assert again.extracted == ("last",)
    assert Shard.exists(directory)


def test_resolution_and_split_get_their_own_shards(dataset) -> None:
    ex = FakeExtractor()
    base = extract_to_cache(**dataset, extractor=ex, layers=["last"], split="train")
    other_res = extract_to_cache(
        **{**dataset, "preprocess": PreprocessConfig(res=32)},
        extractor=ex,
        layers=["last"],
        split="train",
    )
    test_split = extract_to_cache(
        **dataset, extractor=ex, layers=["last"], split="test"
    )

    dirs = {
        base.directories["last"],
        other_res.directories["last"],
        test_split.directories["last"],
    }
    assert len(dirs) == 3
    assert len(Shard(test_split.directories["last"]).image_ids) == 3


def test_roi_mask_flag_does_not_change_the_cache_location(dataset) -> None:
    ex = FakeExtractor()
    on = extract_to_cache(**dataset, extractor=ex, layers=["last"], split="train")
    off = extract_to_cache(
        **{**dataset, "preprocess": PreprocessConfig(res=RES, roi_mask=False)},
        extractor=ex,
        layers=["last"],
        split="train",
    )

    assert on.directories == off.directories
    assert off.extracted == ()


def test_unknown_category_or_split_raises(dataset) -> None:
    with pytest.raises(ValueError):
        extract_to_cache(
            **{**dataset, "category": "nope"},
            extractor=FakeExtractor(),
            layers=["last"],
            split="train",
        )
