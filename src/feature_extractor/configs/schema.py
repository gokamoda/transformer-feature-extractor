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
            "residual.layer_00.pre_attn",
            "residual.layer_00.post_ffn",
        ]
    )
    output_dir: str = "outputs/features"
    save_format: str = "pt"
    batch_size: int = 8
