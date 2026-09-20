from pathlib import Path

import pandas as pd
from PIL import Image

from cmad.data.ingest.mvtec_ad import build_manifest


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3)).save(path)


def _build_fake_dataset(raw_root: Path) -> None:
    root = raw_root / "mvtec_ad"

    cat_a = root / "cat_a"
    for i in range(3):
        _write_png(cat_a / "train" / "good" / f"{i:03d}.png")
    for i in range(2):
        _write_png(cat_a / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        _write_png(cat_a / "test" / "defect1" / f"{i:03d}.png")
        _write_png(cat_a / "ground_truth" / "defect1" / f"{i:03d}_mask.png")
    # defect2: two test images, but only one has a mask -- exercise the
    # missing-mask fallback path.
    _write_png(cat_a / "test" / "defect2" / "000.png")
    _write_png(cat_a / "ground_truth" / "defect2" / "000_mask.png")
    _write_png(cat_a / "test" / "defect2" / "001.png")

    cat_b = root / "cat_b"
    _write_png(cat_b / "train" / "good" / "000.png")
    _write_png(cat_b / "test" / "good" / "000.png")


def test_mvtec_ad_row_counts_and_labels(tmp_path: Path) -> None:
    _build_fake_dataset(tmp_path)
    df = build_manifest(tmp_path)

    assert len(df) == 11
    counts = df.groupby("category").size().to_dict()
    assert counts == {"cat_a": 9, "cat_b": 2}

    assert (df["dataset"] == "mvtec_ad").all()
    assert df["condition"].isna().all()
    assert df["image_id"].is_unique

    label_counts = df["label"].value_counts().to_dict()
    assert label_counts == {0: 7, 1: 4}

    anomalous = df[df["label"] == 1]
    assert (anomalous["anomaly_kind"] == "structural").all()
    normal = df[df["label"] == 0]
    assert (normal["anomaly_kind"] == "none").all()


def test_mvtec_ad_mask_paths(tmp_path: Path) -> None:
    _build_fake_dataset(tmp_path)
    df = build_manifest(tmp_path)

    defect2 = df[(df["category"] == "cat_a") & (df["defect_type"] == "defect2")]
    with_mask = defect2[defect2["image_path"].str.endswith("000.png")]
    without_mask = defect2[defect2["image_path"].str.endswith("001.png")]

    assert with_mask.iloc[0]["mask_path"] == "mvtec_ad/cat_a/ground_truth/defect2/000_mask.png"
    assert pd.isna(without_mask.iloc[0]["mask_path"])

    good = df[(df["category"] == "cat_a") & (df["defect_type"] == "good")]
    assert good["mask_path"].isna().all()
