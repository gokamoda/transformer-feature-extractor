import re
from dataclasses import dataclass, field


@dataclass
class DebugConfig:
    """Configuration for the transformer model."""

    message: str = "Hello, world!"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class FeatureConfig:
    feature_names: list[str] = field(
        default_factory=lambda: [
            "embeddings",
            "layers.layer_00.output",
        ]
    )
    output_dir: str = "outputs/features"
    save_format: str = "pt"
    batch_size: int = 8

    def __post_init__(self):

        patterns = [
            r"^embeddings$",
            r"^layers\.layer_\d{2}\.output$",
        ]

        for feature_name in self.feature_names:
            if not any(re.match(pattern, feature_name) for pattern in patterns):
                raise ValueError(
                    f"Invalid feature name: {feature_name}. "
                    f"Must match one of the following patterns: {patterns}"
                )
