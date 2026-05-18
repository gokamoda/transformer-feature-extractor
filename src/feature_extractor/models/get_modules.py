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


def get_v_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel,
) -> torch.nn.Linear:
    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)

    if architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
        assert architecture.attn_v_proj_field is not None, (
            "Must specify attn_v_proj_field in architecture to get v_proj module."
        )
        v_proj_module = getattr(attn_module, architecture.attn_v_proj_field)
    elif architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
        assert architecture.attn_qkv_proj_field is not None, (
            "Must specify attn_qkv_proj_field in architecture when using conv1d qkv implementation."
        )
        qkv_proj_module = getattr(attn_module, architecture.attn_qkv_proj_field)

        in_features = qkv_proj_module.nx
        out_features = (
            qkv_proj_module.nf // 3
        )  # Assuming q, k, v are concatenated along the output dimension
        if hasattr(qkv_proj_module, "bias") and qkv_proj_module.bias is not None:
            has_bias = True
        else:
            has_bias = False

        v_proj_module = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        )
        v_proj_module.weight = torch.nn.Parameter(
            qkv_proj_module.weight[
                :, 2 * out_features : 3 * out_features
            ].T.contiguous()
        )
        if has_bias:
            v_proj_module.bias = torch.nn.Parameter(
                qkv_proj_module.bias[2 * out_features : 3 * out_features]
            )
    return v_proj_module
