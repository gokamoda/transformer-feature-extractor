from dataclasses import dataclass, field
from typing import Literal

from .architecture import (
    MLP_IMPLEMENTATION_STANDARD,
    QKV_IMPLEMENTATION_CONV1D,
    BaseModelArchitecture,
)


@dataclass
class GPT2Architecture(BaseModelArchitecture):
    config_num_layers: str = "n_layer"
    config_num_attention_heads: str = "n_head"
    config_num_key_value_heads: str = "n_head"
    config_hidden_size: str = "n_embd"
    config_intermediate_size: str = "n_inner"

    supports_layer_output: bool = True
    supports_attention_qkv: bool = True
    supports_mlp_output: bool = True

    model_field: str = "transformer"
    word_embedding_field: str = "wte"
    absolute_pos_embedding_field: str = "wpe"

    layers_field: str = "h"  # field name for transformer blocks in GPT-2
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

    # GPT-2 block internals
    attn_field: str = "attn"
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
    attn_use_rope: bool = False
    attn_position_embeddings_arg_name: str | None = None
    attn_attention_mask_arg_name: str | None = "attention_mask"
    attn_return_fields: list[str] = field(
        default_factory=lambda: ["attn_output", "present", "attn_weights"]
    )
    attn_qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_CONV1D
    )
    attn_q_proj_field: str | None = None
    attn_k_proj_field: str | None = None
    attn_v_proj_field: str | None = None
    attn_qkv_proj_field: str | None = (
        "c_attn"  # Only used if attn_qkv_implementation is "conv1d"
    )
    attn_o_proj_field: str = "c_proj"

    mlp_field: str = "mlp"
    mlp_pos_args: list[str] = field(default_factory=lambda: ["hidden_states"])
    mlp_return_fields: list[str] = field(default_factory=lambda: ["mlp_output"])
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_STANDARD
    mlp_activation_field: str = "act"
    mlp_up_proj_field: str = "c_fc"
    mlp_down_proj_field: str = "c_proj"
