from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from . import mvtec_ad, mvtec_ad2

BUILDERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "mvtec_ad": mvtec_ad.build_manifest,
    "mvtec_ad2": mvtec_ad2.build_manifest,
}
