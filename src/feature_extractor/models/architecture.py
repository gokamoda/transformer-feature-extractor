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

    config_num_layers: str = "num_hidden_layers"

    model_field: str = "model"
    word_embedding_field: str = "embed_tokens"
    absolute_pos_embedding_field: str = None  # default to RoPE

    layers_field: str = "layers"
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
    attn_return_fields: list[str] = field(default_factory=lambda: ["attn_output", "attn_weights"])
    attn_qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
    )
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

@dataclass
class GPT2Architecture:

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
    attn_qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
    )
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


def get_model_architecture(model_class_name: str) -> BaseModelArchitecture:
    """
    Return a BaseModelArchitecture with appropriate field names for the given model class name.
    """
    if "LlamaForCausalLM" in model_class_name:
        return BaseModelArchitecture()
    elif "GPT2LMHeadModel" in model_class_name:
        return GPT2Architecture()
    else:
        logger.warning(
            f"Model class name {model_class_name} not recognized. Using default architecture."
        )
        return BaseModelArchitecture()
