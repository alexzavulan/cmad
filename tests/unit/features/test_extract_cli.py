from pathlib import Path

import numpy as np
import pandas as pd
from _fakes import FakeExtractor
from PIL import Image

from cmad.features import __main__ as cli
from cmad.features.cache import Shard


def test_extract_cli_caches_every_split_of_every_tier_category(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_root = tmp_path / "data"
    rows = []
    for category in ("cat_a", "cat_b"):
        for split in ("train", "test"):
            for i in range(2):
                rel = f"ds/{category}/{split}/{i}.png"
                path = data_root / "raw" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((20, 30, 3), 40 * i + 10, np.uint8)).save(path)
                rows.append(
                    {
                        "image_id": rel,
                        "image_path": rel,
                        "category": category,
                        "orig_split": split,
                        "label": 0,
                        "defect_type": "good",
                    }
                )
    (data_root / "manifests").mkdir()
    pd.DataFrame(rows).to_parquet(data_root / "manifests" / "ds.parquet")

    configs = tmp_path / "configs"
    (configs / "tier").mkdir(parents=True)
    (configs / "tier" / "tiny.yaml").write_text("dataset: ds\ncategories: [cat_a]\n")

    monkeypatch.setattr(cli, "build_extractor", lambda name, device: FakeExtractor())
    argv = [
        "extract", "--backbone", "fake", "--layers", "last", "--tier", "tiny",
        "--res", "16", "--device", "cpu",
        "--data-root", str(data_root), "--configs-dir", str(configs),
    ]  # fmt: skip

    cli.main(argv)
    cli.main(argv)
    cli.main(
        [*argv, "--resize", "20"]
    )  # a different protocol must not reuse the first shards

    out = capsys.readouterr().out
    assert out.count("cat_a/train: extracted") == 2  # first run + the --resize 20 run
    assert out.count("cat_a/test: extracted") == 2
    assert out.count("cache hit") == 2  # second run only
    cached = sorted((data_root / "features" / "fake" / "last").glob("*/*/*"))
    assert sorted(p.parent.name for p in cached) == [
        "test",
        "test",
        "train",
        "train",
    ]  # cat_b never touched
    assert all(Shard.exists(p) for p in cached)
