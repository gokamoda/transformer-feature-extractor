import math

from transformers import PreTrainedConfig

from feature_extractor.models.architecture import BaseModelArchitecture


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
    if architecture.config_head_dim is not None:
        return getattr(model_config, architecture.config_head_dim)
    hidden_size = get_hidden_size(model_config, architecture)
    num_attn_heads = get_num_attn_heads(model_config, architecture)
    return hidden_size // num_attn_heads


def get_kv_hidden_size(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    return get_hidden_size_per_head(model_config, architecture) * get_num_kv_heads(
        model_config, architecture
    )


def get_intermediate_size(
    model_config: PreTrainedConfig, architecture: BaseModelArchitecture
) -> int:
    return getattr(model_config, architecture.config_intermediate_size)


def get_attn_scale(
    model_config: PreTrainedConfig,
    architecture: BaseModelArchitecture,
    head_dim: int,
) -> float:
    """Return the multiplicative factor attention scores are scaled by.

    Most architectures scale by 1/sqrt(head_dim). Gemma2/Gemma3 instead scale
    by 1/sqrt(query_pre_attn_scalar), which need not equal head_dim (see
    `BaseModelArchitecture.attn_scaling_field`).
    """
    if architecture.attn_scaling_field is None:
        return 1.0 / math.sqrt(head_dim)
    return 1.0 / math.sqrt(getattr(model_config, architecture.attn_scaling_field))
