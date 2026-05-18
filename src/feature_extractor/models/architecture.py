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

    # config attribute names
    config_num_layers: str = "num_hidden_layers"
    config_num_attention_heads: str = "num_attention_heads"
    config_num_key_value_heads: str = "num_key_value_heads"
    config_hidden_size: str = "hidden_size"
    config_intermediate_size: str = "intermediate_size"
    attn_use_rope: bool = True
    attn_qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
    )
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_GATED

    # supports
    supports_layer_output: bool = True
    supports_attention_qkv: bool = False
    supports_mlp_output: bool = False

    # level 0
    model_field: str = "model"
    lm_head_field: str = "lm_head"

    # level 1 (inside model)
    word_embedding_field: str = "embed_tokens"
    absolute_pos_embedding_field: str | None = None  # default to RoPE
    rope_field: str | None = "rotary_emb"
    layers_field: str = "layers"
    ln_f_field: str | None = "ln_f"

    # level 2 (inside each layer)
    attn_field: str = "self_attn"
    mlp_field: str = "mlp"
    pre_attn_ln_field: str | None = "input_layernorm"
    pre_mlp_ln_field: str | None = "post_attention_layernorm"

    # level 3 (inside attention module)
    attn_q_proj_field: str | None = "q_proj"
    attn_k_proj_field: str | None = "k_proj"
    attn_v_proj_field: str | None = "v_proj"
    attn_qkv_proj_field: str | None = (
        None  # Only used if attn_qkv_implementation is "conv1d"
    )
    attn_o_proj_field: str = "o_proj"

    # level 3 (inside mlp module)
    mlp_activation_field: str = "act_fn"
    mlp_gate_proj_field: str = "gate_proj"  # Only used if mlp_implementation is "gated"
    mlp_up_proj_field: str = "up_proj"
    mlp_down_proj_field: str = "down_proj"

    # In-out for layer module
    layers_pos_args: list[str] = field(
        default_factory=lambda: [
            "hidden_states",
        ]
    )
    layers_input_hidden_state_arg_name: str = "hidden_states"
    layer_return_fields: list[str] = field(
        default_factory=lambda: ["hidden_states_output"]
    )

    # In-out for attention module
    attn_pos_args: list[str] = field(
        default_factory=lambda: [
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "past_key_values",
        ]
    )
    attn_position_embeddings_arg_name: str | None = "position_embeddings"
    attn_attention_mask_arg_name: str | None = "attention_mask"
    attn_return_fields: list[str] = field(
        default_factory=lambda: ["attn_output", "attn_weights"]
    )

    # In-out for mlp module
    mlp_pos_args: list[str] = field(default_factory=lambda: ["hidden_states"])
    mlp_return_fields: list[str] = field(default_factory=lambda: ["mlp_output"])
