from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torchvision.transforms import v2

from .frames import Frame, PreprocessedFrame

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    res: int
    interpolation: str = "bilinear"
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    roi_mask: bool = True

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.transform = v2.Compose(
            [
                v2.Resize(
                    cfg.res,
                    interpolation=v2.InterpolationMode(cfg.interpolation),
                    antialias=True,
                ),
                v2.CenterCrop(cfg.res),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
            ]
        )

    def __call__(self, frame: Frame) -> PreprocessedFrame:
        chw = torch.from_numpy(frame.image).permute(2, 0, 1).contiguous()
        # ROI mask needs backbone features, so it is fitted and attached after extraction (see roi.py)
        return PreprocessedFrame(tensor=self.transform(chw), roi_mask=None)
