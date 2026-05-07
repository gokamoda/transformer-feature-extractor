from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .extractor.extractor import FeatureExtractor
    from .configs import FeatureConfig

__all__ = ["FeatureExtractor", "FeatureConfig"]


def __getattr__(name: str):
    if name == "FeatureExtractor":
        from .extractor.extractor import FeatureExtractor

        return FeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
