from dataclasses import dataclass, field
from typing import Literal

from .architecture import (
    MLP_IMPLEMENTATION_STANDARD,
    QKV_IMPLEMENTATION_CONV1D,
    BaseModelArchitecture,
)


@dataclass
class GPT2Architecture(BaseModelArchitecture):
    # config attribute names
    config_num_layers: str = "n_layer"
    config_num_attention_heads: str = "n_head"
    config_num_key_value_heads: str = "n_head"
    config_hidden_size: str = "n_embd"
    config_intermediate_size: str = "n_inner"
    attn_use_rope: bool = False
    attn_qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_CONV1D
    )
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_STANDARD

    # supports
    supports_layer_output: bool = True
    supports_attention_qkv: bool = True
    supports_mlp_output: bool = True

    # level 0
    model_field: str = "transformer"
    lm_head_field: str = "lm_head"

    # level 1 (inside model)
    word_embedding_field: str = "wte"
    absolute_pos_embedding_field: str = "wpe"
    rope_field: str | None = None
    layers_field: str = "h"  # field name for transformer blocks in GPT-2
    ln_f_field: str = "ln_f"

    # level 2 (inside each layer)
    attn_field: str = "attn"
    mlp_field: str = "mlp"
    pre_attn_ln_field: str = "ln_1"
    pre_mlp_ln_field: str = "ln_2"

    # level 3 (inside attention module)
    attn_q_proj_field: str | None = None
    attn_k_proj_field: str | None = None
    attn_v_proj_field: str | None = None
    attn_qkv_proj_field: str | None = (
        "c_attn"  # Only used if attn_qkv_implementation is "conv1d"
    )
    attn_o_proj_field: str = "c_proj"

    # level 3 (inside mlp module)
    mlp_activation_field: str = "act"
    mlp_up_proj_field: str = "c_fc"
    mlp_down_proj_field: str = "c_proj"

    # In-out for layer module
    layers_pos_args: list[str] = field(
        default_factory=lambda: [
            "hidden_states",
            "past_key_values",
            "cache_position",
            "causal_mask",
            "encoder_hidden_states",
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
            "last_key_values",
            "attention_mask",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "output_attentions",
        ]
    )
    attn_position_embeddings_arg_name: str | None = None
    attn_attention_mask_arg_name: str | None = "attention_mask"
    attn_return_fields: list[str] = field(
        default_factory=lambda: ["attn_output", "present", "attn_weights"]
    )

    # In-out for mlp module
    mlp_pos_args: list[str] = field(default_factory=lambda: ["hidden_states"])
    mlp_return_fields: list[str] = field(default_factory=lambda: ["mlp_output"])
