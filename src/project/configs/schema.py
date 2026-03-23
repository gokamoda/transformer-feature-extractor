from dataclasses import dataclass, field


@dataclass
class DebugConfig:
    """Configuration for the transformer model."""

    message: str = "Hello, world!"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    debug: DebugConfig = field(default_factory=DebugConfig)
