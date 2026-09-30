from .base import FeatureBatch, FeatureExtractor, build_extractor, register
from .dinov2 import DinoV2Extractor

__all__ = [
    "DinoV2Extractor",
    "FeatureBatch",
    "FeatureExtractor",
    "build_extractor",
    "register",
]
