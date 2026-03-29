from dataclasses import dataclass, field
from typing import Literal

from .architecture import (
    MLP_IMPLEMENTATION_STANDARD,
    QKV_IMPLEMENTATION_CONV1D,
    BaseModelArchitecture,
)


@dataclass
class GPTNeoXArchitecture(BaseModelArchitecture):
    supports_layer_output: bool = True
    supports_attention_qkv: bool = True
    supports_mlp_output: bool = True

    model_field: str = "gpt_neox"
    layers_field: str = "layers"

    attn_field: str = "attention"
    attn_qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_CONV1D
    )
    attn_q_proj_field: str | None = None
    attn_k_proj_field: str | None = None
    attn_v_proj_field: str | None = None
    attn_qkv_proj_field: str | None = "query_key_value"
    attn_o_proj_field: str = "dense"

    mlp_field: str = "mlp"
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_STANDARD
    mlp_activation_field: str = "act"
    mlp_up_proj_field: str = "dense_h_to_4h"
    mlp_down_proj_field: str = "dense_4h_to_h"

    layer_return_fields: list[str] = field(default_factory=lambda: ["hidden_states"])
