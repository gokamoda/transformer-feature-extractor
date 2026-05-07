import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbeddingFeatureSpec:
    module: str = "embeddings"

    def __post_init__(self):
        if self.module != "embeddings":
            raise ValueError(f"Invalid embedding module: {self.module}")

    def to_feature_name(self) -> str:
        return self.module


@dataclass(frozen=True)
class LayerFeatureSpec:
    layer_index: int
    feature: str
    module: str = "layers"

    def __post_init__(self):
        if self.module != "layers":
            raise ValueError(f"Invalid layer module: {self.module}")
        if self.layer_index < 0:
            raise ValueError(
                f"layer_index must be non-negative, got {self.layer_index}"
            )
        if self.feature not in {"input", "output"}:
            raise ValueError(f"Invalid layer feature: {self.feature}")

    def to_feature_name(self) -> str:
        return f"layers.layer_{self.layer_index:02d}.{self.feature}"


@dataclass(frozen=True)
class AttentionFeatureSpec:
    layer_index: int
    feature: str
    module: str = "attn"

    def __post_init__(self):
        if self.module != "attn":
            raise ValueError(f"Invalid attention module: {self.module}")
        if self.layer_index < 0:
            raise ValueError(
                f"layer_index must be non-negative, got {self.layer_index}"
            )
        if self.feature not in {
            "query",
            "key",
            "value",
            "attn_weights",
            "output",
            "attention_mask",
            "positional_embedding",
        }:
            raise ValueError(f"Invalid attention feature: {self.feature}")

    def to_feature_name(self) -> str:
        return f"attn.layer_{self.layer_index:02d}.{self.feature}"


@dataclass(frozen=True)
class MLPFeatureSpec:
    layer_index: int
    feature: str
    module: str = "mlp"

    def __post_init__(self):
        if self.module != "mlp":
            raise ValueError(f"Invalid MLP module: {self.module}")
        if self.layer_index < 0:
            raise ValueError(
                f"layer_index must be non-negative, got {self.layer_index}"
            )
        if self.feature not in {"activation", "down_proj_input", "output"}:
            raise ValueError(f"Invalid MLP feature: {self.feature}")

    def to_feature_name(self) -> str:
        return f"mlp.layer_{self.layer_index:02d}.{self.feature}"


FeatureSpec = (
    EmbeddingFeatureSpec | LayerFeatureSpec | AttentionFeatureSpec | MLPFeatureSpec
)


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
    feature_specs: list[FeatureSpec] = field(
        default_factory=lambda: [
            EmbeddingFeatureSpec(),
            LayerFeatureSpec(layer_index=0, feature="output"),
        ]
    )
    batch_size: int = 8

    @classmethod
    def from_str(cls, feature_names: list[str], batch_size: int = 8) -> "FeatureConfig":
        feature_specs = []
        embedding_pattern = re.compile(r"^embeddings$")
        layer_pattern = re.compile(r"^layers\.layer_(\d+)\.(input|output)$")
        attn_pattern = re.compile(
            r"^attn\.layer_(\d+)\.(query|key|value|attn_weights|output|attention_mask|positional_embedding)$"
        )
        mlp_pattern = re.compile(
            r"^mlp\.layer_(\d+)\.(activation|down_proj_input|output)$"
        )
        patterns = {
            "embeddings": embedding_pattern,
            "layers": layer_pattern,
            "attn": attn_pattern,
            "mlp": mlp_pattern,
        }
        for feature in feature_names:
            matched = False
            for module_name, pattern in patterns.items():
                match = pattern.match(feature)
                if not match:
                    continue

                if module_name == "embeddings":
                    feature_specs.append(EmbeddingFeatureSpec())
                    matched = True
                    break

                layer_index = int(match.group(1))
                spec_feature = match.group(2)
                if module_name == "layers":
                    feature_specs.append(
                        LayerFeatureSpec(layer_index=layer_index, feature=spec_feature)
                    )
                    matched = True
                    break
                if module_name == "attn":
                    feature_specs.append(
                        AttentionFeatureSpec(
                            layer_index=layer_index, feature=spec_feature
                        )
                    )
                    matched = True
                    break
                if module_name == "mlp":
                    feature_specs.append(
                        MLPFeatureSpec(layer_index=layer_index, feature=spec_feature)
                    )
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Invalid feature name: {feature}. Must match one of the supported feature spec formats."
                )

        return cls(feature_specs=feature_specs, batch_size=batch_size)

    def feature_names_as_strings(self) -> list[str]:
        return [feature.to_feature_name() for feature in self.feature_specs]

    def deepest_layer_index(self) -> int | None:
        layer_indices = []
        for feature in self.feature_specs:
            if isinstance(
                feature, (LayerFeatureSpec, AttentionFeatureSpec, MLPFeatureSpec)
            ):
                layer_indices.append(feature.layer_index)
        return max(layer_indices) if layer_indices else None
