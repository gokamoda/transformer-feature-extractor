from dataclasses import dataclass

from .architecture import BaseModelArchitecture


@dataclass
class Qwen3Architecture(BaseModelArchitecture):
    supports_layer_output: bool = True
    supports_attention_qkv: bool = True
    supports_mlp_output: bool = True

    attn_q_norm_field: str | None = "q_norm"
    attn_k_norm_field: str | None = "k_norm"

    # Qwen3's head_dim need not equal hidden_size // num_attention_heads.
    config_head_dim: str | None = "head_dim"
