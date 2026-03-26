from dataclasses import dataclass, field
from typing import Literal

from .base_architecture import (
    MLP_IMPLEMENTATION_GATED,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
    BaseModelArchitecture,
)


@dataclass
class GPT2Architecture(BaseModelArchitecture):
    config_num_layers: str = "n_layer"

    model_field: str = "transformer"
    word_embedding_field: str = "wte"
    absolute_pos_embedding_field: str = "wpe"

    layers_field: str = "h"  # "h" is the field name for transformer blocks in GPT-2
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
    attn_qkv_implementation = QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
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
