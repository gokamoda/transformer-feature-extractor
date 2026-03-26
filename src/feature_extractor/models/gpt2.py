from dataclasses import dataclass, field
from typing import Literal

from .base_architecture import (
    MLP_IMPLEMENTATION_STANDARD,
    QKV_IMPLEMENTATION_CONV1D,
    BaseModelArchitecture,
)


@dataclass
class GPT2Architecture(BaseModelArchitecture):
    config_num_layers: str = "n_layer"

    supports_layer_output: bool = True
    supports_attention_qkv: bool = True
    supports_mlp_output: bool = True

    model_field: str = "transformer"
    word_embedding_field: str = "wte"
    absolute_pos_embedding_field: str = "wpe"

    layers_field: str = "h"  # field name for transformer blocks in GPT-2
    layer_return_fields: list[str] = field(default_factory=lambda: ["hidden_states"])

    # GPT-2 block internals
    attn_field: str = "attn"
    attn_pos_args: list[str] = field(
        default_factory=lambda: [
            "hidden_states",
            "past_key_values",
            "attention_mask",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "output_attentions"
        ]
    )
    attn_return_fields: list[str] = field(
        default_factory=lambda: ["attn_output", "attn_weights"]
    )
    attn_qkv_implementation = QKV_IMPLEMENTATION_CONV1D
    attn_conv1d_field: str = "c_attn"
    attn_o_proj_field: str = "c_proj"

    mlp_field: str = "mlp"
    mlp_pos_args: list[str] = field(default_factory=lambda: ["hidden_states"])
    mlp_return_fields: list[str] = field(default_factory=lambda: ["mlp_output"])
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_STANDARD
    mlp_up_proj_field: str = "c_fc"
    mlp_down_proj_field: str = "c_proj"
