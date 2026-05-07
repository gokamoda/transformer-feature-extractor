from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configs import FeatureConfig
    from .extractor.extractor import FeatureExtractor

__all__ = ["FeatureExtractor", "FeatureConfig"]


def __getattr__(name: str):
    if name == "FeatureExtractor":
        from .extractor.extractor import FeatureExtractor

        return FeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
