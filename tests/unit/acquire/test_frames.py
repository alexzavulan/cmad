from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cmad.acquire import ManifestSource


def test_manifest_source_yields_rgb_frames_with_meta(tmp_path: Path) -> None:
    (tmp_path / "ds").mkdir()
    Image.new("RGB", (6, 4), (10, 20, 30)).save(tmp_path / "ds" / "a.png")
    Image.new("L", (5, 7), 128).save(tmp_path / "ds" / "b.png")
    manifest = pd.DataFrame(
        {
            "image_id": ["ds/a.png", "ds/b.png"],
            "image_path": ["ds/a.png", "ds/b.png"],
            "category": ["bottle", "screw"],
            "label": [0, 1],
            "defect_type": ["good", "scratch"],
        }
    )

    frames = list(ManifestSource(manifest, tmp_path))

    assert len(frames) == 2
    assert frames[0].image.shape == (4, 6, 3)
    assert frames[0].image.dtype == np.uint8
    assert tuple(frames[0].image[0, 0]) == (10, 20, 30)
    assert frames[1].image.shape == (7, 5, 3)  # grayscale lifted to 3 channels
    assert frames[1].meta == {
        "image_id": "ds/b.png",
        "category": "screw",
        "label": 1,
        "defect_type": "scratch",
    }
