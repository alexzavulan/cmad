from pathlib import Path

import pandas as pd
from PIL import Image

from cmad.data.ingest.mvtec_ad2 import build_manifest


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3)).save(path)


def _build_fake_dataset(raw_root: Path) -> None:
    cat = raw_root / "mvtec_ad2" / "toy"

    _write_png(cat / "train" / "good" / "000_regular.png")
    _write_png(cat / "train" / "good" / "001_regular.png")
    _write_png(cat / "validation" / "good" / "000_regular.png")

    _write_png(cat / "test_public" / "good" / "000_regular.png")
    _write_png(cat / "test_public" / "good" / "000_shift_1.png")

    _write_png(cat / "test_public" / "bad" / "001_regular.png")
    _write_png(cat / "test_public" / "ground_truth" / "bad" / "001_regular_mask.png")
    # overexposed bad image deliberately has no mask -- missing-mask fallback.
    _write_png(cat / "test_public" / "bad" / "001_overexposed.png")

    # server-evaluated splits: must never show up in the manifest.
    _write_png(cat / "test_private" / "999_regular.png")
    _write_png(cat / "test_private_mixed" / "999_regular.png")


def test_mvtec_ad2_skips_private_splits(tmp_path: Path) -> None:
    _build_fake_dataset(tmp_path)
    df = build_manifest(tmp_path)

    assert len(df) == 7
    assert set(df["orig_split"]) == {"train", "validation", "test_public"}
    assert not df["image_path"].str.contains("test_private").any()


def test_mvtec_ad2_labels_and_conditions(tmp_path: Path) -> None:
    _build_fake_dataset(tmp_path)
    df = build_manifest(tmp_path)

    assert (df["dataset"] == "mvtec_ad2").all()
    assert (df["category"] == "toy").all()

    label_counts = df["label"].value_counts().to_dict()
    assert label_counts == {0: 5, 1: 2}

    good = df[df["defect_type"] == "good"]
    assert (good["label"] == 0).all()
    assert (good["anomaly_kind"] == "none").all()

    bad = df[df["defect_type"] == "bad"]
    assert (bad["label"] == 1).all()
    assert (bad["anomaly_kind"] == "structural").all()

    test_public_good = df[(df["orig_split"] == "test_public") & (df["defect_type"] == "good")]
    assert set(test_public_good["condition"]) == {"regular", "shift_1"}


def test_mvtec_ad2_mask_paths(tmp_path: Path) -> None:
    _build_fake_dataset(tmp_path)
    df = build_manifest(tmp_path)

    bad = df[df["defect_type"] == "bad"]
    with_mask = bad[bad["condition"] == "regular"]
    without_mask = bad[bad["condition"] == "overexposed"]

    assert with_mask.iloc[0]["mask_path"] == (
        "mvtec_ad2/toy/test_public/ground_truth/bad/001_regular_mask.png"
    )
    assert pd.isna(without_mask.iloc[0]["mask_path"])


def test_item_id_groups_conditions_and_never_collides_across_folders(tmp_path: Path) -> None:
    cat = tmp_path / "mvtec_ad2" / "toy"
    for folder in ("train/good", "validation/good", "test_public/good", "test_public/bad"):
        for condition in ("regular", "overexposed") if folder.startswith("test_public") else ("regular",):
            _write_png(cat / folder / f"000_{condition}.png")  # every folder counts from 000
    df = build_manifest(tmp_path)

    assert len(df) == 6
    # the numeric prefix is the same everywhere, yet the four folders are four different items
    assert df["item_id"].nunique() == 4
    # the lighting variants of one test_public item share one item_id
    bad = df[(df["orig_split"] == "test_public") & (df["label"] == 1)]
    assert len(bad) == 2 and bad["item_id"].nunique() == 1
    assert bad["item_id"].iloc[0] == "mvtec_ad2/toy/test_public/bad/000"
    # train and validation never share an item
    train_items = set(df[df["orig_split"] == "train"]["item_id"])
    val_items = set(df[df["orig_split"] == "validation"]["item_id"])
    assert not train_items & val_items
