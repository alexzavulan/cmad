import numpy as np
import pandas as pd
import torch
from _fakes import FakeExtractor
from PIL import Image

from cmad.acquire import PreprocessConfig, rotate_image
from cmad.features import extract_rotated

CFG = PreprocessConfig(res=16)


def _rows(tmp_path, n=3):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rel = f"ds/cat/train/{i}.png"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)).save(
            tmp_path / rel
        )
        rows.append(
            {
                "image_id": rel,
                "image_path": rel,
                "category": "cat",
                "label": 0,
                "defect_type": "good",
            }
        )
    return pd.DataFrame(rows)


def test_rotated_features_are_image_major_and_angle_zero_is_the_plain_image(
    tmp_path,
) -> None:
    rows = _rows(tmp_path)
    angles = (0, 90, 180)

    out = extract_rotated(
        FakeExtractor(), rows, tmp_path, CFG, ["last"], angles, batch_size=4
    )

    assert out.patches["last"].shape == (3 * 3, 4, 3)  # 3 images x 3 angles, a 2x2 grid
    plain = extract_rotated(FakeExtractor(), rows, tmp_path, CFG, ["last"], (0,))
    assert torch.equal(
        out.patches["last"][::3], plain.patches["last"]
    )  # angle 0 of each image
    assert not torch.equal(
        out.patches["last"][1], out.patches["last"][0]
    )  # 90 degrees differs


def test_batch_size_does_not_change_the_features(tmp_path) -> None:
    rows = _rows(tmp_path)

    a = extract_rotated(
        FakeExtractor(), rows, tmp_path, CFG, ["last"], (0, 45), batch_size=1
    )
    b = extract_rotated(
        FakeExtractor(), rows, tmp_path, CFG, ["last"], (0, 45), batch_size=16
    )

    assert torch.equal(a.patches["last"], b.patches["last"])


def test_the_rotation_happens_before_the_resize(tmp_path) -> None:
    rows = _rows(tmp_path, n=1)
    img = np.array(Image.open(tmp_path / rows.image_path.iloc[0]).convert("RGB"))

    from cmad.acquire import Frame, Preprocessor

    expected = Preprocessor(CFG)(Frame(image=rotate_image(img, 45))).tensor
    got = FakeExtractor().extract(expected[None], ["last"]).patches["last"]
    out = extract_rotated(FakeExtractor(), rows, tmp_path, CFG, ["last"], (45,))

    assert torch.equal(out.patches["last"], got)
