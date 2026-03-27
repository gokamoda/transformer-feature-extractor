from dataclasses import dataclass, field
from typing import Literal

from transformers import PreTrainedConfig

from feature_extractor.logger import init_logging

logger = init_logging(__name__)

MLP_IMPLEMENTATION_STANDARD = "standard"
MLP_IMPLEMENTATION_GATED = "gated"
QKV_IMPLEMENTATION_INDEPENDENT_LINEAR = "independent_linear"
QKV_IMPLEMENTATION_CONV1D = "conv1d"


@dataclass
class BaseModelArchitecture:
    """
    Describe model attribute names for feature extraction hooks.
    Defaults to llama-3.2
    """

    config_num_layers: str = "num_hidden_layers"
    config_num_attention_heads: str = "num_attention_heads"
    config_num_key_value_heads: str = "num_key_value_heads"
    config_hidden_size: str = "hidden_size"

    # Capability flags. Keep these explicit so callers can fail fast for
    # unsupported feature families.
    supports_layer_output: bool = True
    supports_attention_qkv: bool = False
    supports_mlp_output: bool = False

    model_field: str = "model"
    word_embedding_field: str = "embed_tokens"
    absolute_pos_embedding_field: str | None = None  # default to RoPE

    layers_field: str = "layers"
    layer_return_fields: list[str] = field(default_factory=lambda: ["hidden_states"])

    attn_field: str = "self_attn"
    attn_pos_args: list[str] = field(
        default_factory=lambda: [
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "past_key_values",
        ]
    )
    attn_return_fields: list[str] = field(
        default_factory=lambda: ["attn_output", "attn_weights"]
    )
    attn_qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
    )
    attn_q_proj_field: str | None = "q_proj"
    attn_k_proj_field: str | None = "k_proj"
    attn_v_proj_field: str | None = "v_proj"
    attn_qkv_proj_field: str | None = (
        None  # Only used if attn_qkv_implementation is "conv1d"
    )
    attn_o_proj_field: str = "o_proj"

    mlp_field: str = "mlp"
    mlp_pos_args: list[str] = field(default_factory=lambda: ["hidden_states"])
    mlp_return_fields: list[str] = field(default_factory=lambda: ["mlp_output"])
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_GATED
    mlp_gate_proj_field: str = "gate_proj"  # Only used if mlp_implementation is "gated"
    mlp_up_proj_field: str = "up_proj"
    mlp_down_proj_field: str = "down_proj"


def get_num_layers(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    return getattr(model_config, architecture.config_num_layers)


def get_num_attn_heads(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    return getattr(model_config, architecture.config_num_attention_heads)


def get_num_kv_heads(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    if hasattr(model_config, architecture.config_num_key_value_heads):
        return getattr(model_config, architecture.config_num_key_value_heads)
    else:  # If num_key_value_heads is not defined, assume it's the same as num_attention_heads
        return get_num_attn_heads(model_config, architecture)


def get_hidden_size(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    return getattr(model_config, architecture.config_hidden_size)


def get_hidden_size_per_head(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    hidden_size = get_hidden_size(model_config, architecture)
    num_attn_heads = get_num_attn_heads(model_config, architecture)
    return hidden_size // num_attn_heads


def get_kv_hidden_size(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    return get_hidden_size_per_head(model_config, architecture) * get_num_kv_heads(
        model_config, architecture
    )
