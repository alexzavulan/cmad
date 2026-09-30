from .base import FeatureBatch, FeatureExtractor, build_extractor, register
from .dinov2 import DinoV2Extractor
from .wrn import WideResNet50Extractor

__all__ = [
    "DinoV2Extractor",
    "FeatureBatch",
    "FeatureExtractor",
    "WideResNet50Extractor",
    "build_extractor",
    "register",
]
