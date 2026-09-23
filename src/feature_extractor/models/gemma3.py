from dataclasses import dataclass

from .architecture import BaseModelArchitecture


@dataclass
class Gemma3Architecture(BaseModelArchitecture):
    supports_layer_output: bool = True
    supports_attention_qkv: bool = True
    supports_mlp_output: bool = True

    attn_q_norm_field: str | None = "q_norm"
    attn_k_norm_field: str | None = "k_norm"

    # Gemma3's head_dim need not equal hidden_size // num_attention_heads.
    config_head_dim: str | None = "head_dim"

    # Gemma3 scales attention scores by 1/sqrt(query_pre_attn_scalar), which
    # need not equal head_dim (they happen to coincide on some sizes, e.g.
    # gemma-3-1b-pt, but not necessarily on others).
    attn_scaling_field: str | None = "query_pre_attn_scalar"

    # Gemma3 uses a sandwich norm: input_layernorm -> attn -> post_attention_layernorm
    # -> +residual -> pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm
    # -> +residual. Note that `post_attention_layernorm` here is a *post*-attention
    # norm, unlike the base default where that same HF attribute name is reused as
    # the *pre*-mlp norm.
    pre_mlp_ln_field: str | None = "pre_feedforward_layernorm"
    post_attn_ln_field: str | None = "post_attention_layernorm"
    post_mlp_ln_field: str | None = "post_feedforward_layernorm"

    # model.rotary_emb keeps separate inv_freq/attention_scaling buffers per
    # layer type (e.g. "sliding_attention" vs "full_attention" -> different
    # rope_theta), selected via config.layer_types[layer_index].
    config_layer_types: str | None = "layer_types"
