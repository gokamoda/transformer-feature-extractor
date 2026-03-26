from dataclasses import dataclass, field
from typing import Literal

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
    attn_q_proj_field: str = "q_proj"
    attn_k_proj_field: str = "k_proj"
    attn_v_proj_field: str = "v_proj"
    attn_o_proj_field: str = "o_proj"

    mlp_field: str = "mlp"
    mlp_pos_args: list[str] = field(default_factory=lambda: ["hidden_states"])
    mlp_return_fields: list[str] = field(default_factory=lambda: ["mlp_output"])
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_GATED
    mlp_gate_proj_field: str = "gate_proj"  # Only used if mlp_implementation is "gated"
    mlp_up_proj_field: str = "up_proj"
    mlp_down_proj_field: str = "down_proj"
