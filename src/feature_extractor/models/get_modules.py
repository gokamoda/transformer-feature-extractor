from typing import Literal
import torch
from transformers import PreTrainedModel

from feature_extractor.models.architecture import (
    QKV_IMPLEMENTATION_CONV1D,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
    BaseModelArchitecture,
)


def get_pre_attn_norm_module(
    model: PreTrainedModel,
    architecture: BaseModelArchitecture,
    layer_index: int,
):
    assert architecture.pre_attn_ln_field is not None, (
        "Architecture does not specify a pre-attention layer norm field."
    )
    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    return getattr(layer_module, architecture.pre_attn_ln_field)


def get_qkv_proj_module_gpt2(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
    module: Literal["q_proj", "k_proj", "v_proj"],
):
    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)
    
    assert architecture.attn_qkv_proj_field is not None, (
        "Must specify attn_qkv_proj_field in architecture when using conv1d qkv implementation."
    )
    qkv_proj_module = getattr(attn_module, architecture.attn_qkv_proj_field)

    weight_order = {"q_proj": 0, "k_proj": 1, "v_proj": 2}

    in_features = qkv_proj_module.nx
    out_features = (
        qkv_proj_module.nf // 3
    )  # Assuming q, k, v are concatenated along the output dimension
    if hasattr(qkv_proj_module, "bias") and qkv_proj_module.bias is not None:
        has_bias = True
    else:
        has_bias = False

    proj_module = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
    )

    proj_module.weight = torch.nn.Parameter(
        qkv_proj_module.weight[
            :, weight_order[module] * out_features : (weight_order[module] + 1) * out_features
        ].clone().T.contiguous()
    )
    if has_bias:
        proj_module.bias = torch.nn.Parameter(
            qkv_proj_module.bias[weight_order[module] * out_features : (weight_order[module] + 1) * out_features]
        )
    return proj_module

def get_qkv_proj_module_independent_linear(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
    module: Literal["q_proj", "k_proj", "v_proj"],
):
    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)

    if module == "q_proj":
        assert architecture.attn_q_proj_field is not None, (
            "Architecture does not specify attn_q_proj_field."
        )
        return getattr(attn_module, architecture.attn_q_proj_field)
    elif module == "k_proj":
        assert architecture.attn_k_proj_field is not None, (
            "Architecture does not specify attn_k_proj_field."
        )
        return getattr(attn_module, architecture.attn_k_proj_field)
    elif module == "v_proj":
        assert architecture.attn_v_proj_field is not None, (
            "Architecture does not specify attn_v_proj_field."
        )
        return getattr(attn_module, architecture.attn_v_proj_field)
    else:
        raise ValueError(f"Invalid module name: {module}. Must be one of 'q_proj', 'k_proj', 'v_proj'.")


def get_o_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
):
    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)
    o_proj_module = getattr(attn_module, architecture.attn_o_proj_field)
    return o_proj_module


def _get_qkv_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
    module: Literal["q_proj", "k_proj", "v_proj"],      
):
    if architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
        return get_qkv_proj_module_independent_linear(
            architecture=architecture,
            layer_index=layer_index,
            model=model,
            module=module,
        )
    elif architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
        return get_qkv_proj_module_gpt2(
            architecture=architecture,
            layer_index=layer_index,
            model=model,
            module=module,
        )
    else:
        raise ValueError(f"Unsupported attn_qkv_implementation: {architecture.attn_qkv_implementation}")


def get_v_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
) -> torch.nn.Linear:
    return _get_qkv_proj_module(
        architecture=architecture,
        layer_index=layer_index,
        model=model,
        module="v_proj",
    )

def get_q_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
) -> torch.nn.Linear:
    return _get_qkv_proj_module(
        architecture=architecture,
        layer_index=layer_index,
        model=model,
        module="q_proj",
    )

def get_k_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
) -> torch.nn.Linear:
    return _get_qkv_proj_module(
        architecture=architecture,
        layer_index=layer_index,
        model=model,
        module="k_proj",
    )


def get_rope_module(
    architecture: BaseModelArchitecture,
    model: PreTrainedModel,
) -> torch.nn.Module:
    model_module = getattr(model, architecture.model_field)
    return getattr(model_module, architecture.rope_field)
