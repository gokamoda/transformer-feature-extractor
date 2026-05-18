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
