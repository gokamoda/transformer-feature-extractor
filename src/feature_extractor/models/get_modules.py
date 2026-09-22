import copy
import gc
from typing import Literal

import torch
from transformers import PreTrainedConfig, PreTrainedModel

from feature_extractor.models.architecture import (
    QKV_IMPLEMENTATION_CONV1D,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
    BaseModelArchitecture,
)

from .load import load_causal_model


def get_pre_attn_norm_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    assert architecture.pre_attn_ln_field is not None, (
        "Architecture does not specify a pre-attention layer norm field."
    )

    if model is None:
        load_model_inside_function = True
    else:
        load_model_inside_function = False

    if load_model_inside_function:
        assert model_name is not None, "model_name must be provided if model is None"
        model = load_causal_model(model_name, device=device)

    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    pre_attn_norm_module = copy.deepcopy(
        getattr(layer_module, architecture.pre_attn_ln_field)
    )

    if load_model_inside_function:
        del layer_module
        del model_module
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return pre_attn_norm_module


def _get_attn_norm_module(
    architecture: BaseModelArchitecture,
    norm_field: str | None,
    field_name: str,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    assert norm_field is not None, (
        f"Architecture does not specify a {field_name}."
    )

    if model is None:
        load_model_inside_function = True
    else:
        load_model_inside_function = False

    if load_model_inside_function:
        assert model_name is not None, "model_name must be provided if model is None"
        model = load_causal_model(model_name, device=device)

    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)
    norm_module = copy.deepcopy(getattr(attn_module, norm_field))

    if load_model_inside_function:
        del attn_module
        del layer_module
        del model_module
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return norm_module


def get_q_norm_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """Return the RMSNorm applied to q_proj output (before RoPE), if any.

    Only defined for architectures that set `attn_q_norm_field` (e.g. Qwen3,
    Gemma3). Raises if the architecture does not define one.
    """
    return _get_attn_norm_module(
        architecture=architecture,
        norm_field=architecture.attn_q_norm_field,
        field_name="attn_q_norm_field",
        layer_index=layer_index,
        model=model,
        model_name=model_name,
        device=device,
    )


def get_k_norm_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> torch.nn.Module:
    """Return the RMSNorm applied to k_proj output (before RoPE), if any.

    Only defined for architectures that set `attn_k_norm_field` (e.g. Qwen3,
    Gemma3). Raises if the architecture does not define one.
    """
    return _get_attn_norm_module(
        architecture=architecture,
        norm_field=architecture.attn_k_norm_field,
        field_name="attn_k_norm_field",
        layer_index=layer_index,
        model=model,
        model_name=model_name,
        device=device,
    )


def get_qkv_proj_module_gpt2(
    architecture: BaseModelArchitecture,
    layer_index: int,
    modules: list[Literal["q_proj", "k_proj", "v_proj"]],
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
    device: str | None = None,
):
    if model is None:
        load_model_inside_function = True
    else:
        load_model_inside_function = False

    if load_model_inside_function:
        assert model_name is not None, "model_name must be provided if model is None"
        model = load_causal_model(model_name, device=device)

    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)

    assert architecture.attn_qkv_proj_field is not None, (
        "Must specify attn_qkv_proj_field in architecture when using conv1d qkv implementation."
    )

    weight_order = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
    qkv_proj_module: torch.nn.Module = getattr(
        attn_module, architecture.attn_qkv_proj_field
    )
    in_features = qkv_proj_module.nx
    out_features = (
        qkv_proj_module.nf // 3
    )  # Assuming q, k, v are concatenated along the output dimension
    if hasattr(qkv_proj_module, "bias") and qkv_proj_module.bias is not None:
        has_bias = True
    else:
        has_bias = False

    retrieved_modules = {}
    for module_name in modules:
        proj_module = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        )

        proj_module.weight = torch.nn.Parameter(
            qkv_proj_module.weight[
                :,
                weight_order[module_name] * out_features : (
                    weight_order[module_name] + 1
                )
                * out_features,
            ]
            .clone()
            .T.contiguous()
        )
        if has_bias:
            proj_module.bias = torch.nn.Parameter(
                qkv_proj_module.bias[
                    weight_order[module_name] * out_features : (
                        weight_order[module_name] + 1
                    )
                    * out_features
                ]
                .clone()
                .contiguous()
            )
        retrieved_modules[module_name] = proj_module

    if load_model_inside_function:
        del attn_module
        del layer_module
        del model_module
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return retrieved_modules


def get_qkv_proj_module_independent_linear(
    architecture: BaseModelArchitecture,
    layer_index: int,
    modules: list[Literal["q_proj", "k_proj", "v_proj"]],
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
    device: str | None = None,
):
    if model is None:
        load_model_inside_function = True
    else:
        load_model_inside_function = False

    if load_model_inside_function:
        assert model_name is not None, "model_name must be provided if model is None"
        model = load_causal_model(model_name, device=device)

    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)

    retrieved_modules = {}

    if "q_proj" in modules:
        assert architecture.attn_q_proj_field is not None, (
            "Architecture does not specify attn_q_proj_field."
        )
        retrieved_modules["q_proj"] = copy.deepcopy(
            getattr(attn_module, architecture.attn_q_proj_field)
        )

    if "k_proj" in modules:
        assert architecture.attn_k_proj_field is not None, (
            "Architecture does not specify attn_k_proj_field."
        )
        retrieved_modules["k_proj"] = copy.deepcopy(
            getattr(attn_module, architecture.attn_k_proj_field)
        )

    if "v_proj" in modules:
        assert architecture.attn_v_proj_field is not None, (
            "Architecture does not specify attn_v_proj_field."
        )
        retrieved_modules["v_proj"] = copy.deepcopy(
            getattr(attn_module, architecture.attn_v_proj_field)
        )
    if load_model_inside_function:
        del attn_module
        del layer_module
        del model_module
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return retrieved_modules


def get_o_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
):
    if model is None:
        load_model_inside_function = True
    else:
        load_model_inside_function = False

    if load_model_inside_function:
        assert model_name is not None, "model_name must be provided if model is None"
        model = load_causal_model(model_name)

    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)
    o_proj_module = copy.deepcopy(getattr(attn_module, architecture.attn_o_proj_field))

    print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
    print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")
    if load_model_inside_function:
        del attn_module
        del layer_module
        del model_module
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
    print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")

    return o_proj_module


def _get_qkv_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    modules: list[Literal["q_proj", "k_proj", "v_proj"]],
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
):
    if architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
        return get_qkv_proj_module_independent_linear(
            architecture=architecture,
            layer_index=layer_index,
            model=model,
            model_name=model_name,
            modules=modules,
        )
    elif architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
        return get_qkv_proj_module_gpt2(
            architecture=architecture,
            layer_index=layer_index,
            model=model,
            model_name=model_name,
            modules=modules,
        )
    else:
        raise ValueError(
            f"Unsupported attn_qkv_implementation: {architecture.attn_qkv_implementation}"
        )


def get_v_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
) -> torch.nn.Linear:
    return _get_qkv_proj_module(
        architecture=architecture,
        layer_index=layer_index,
        model=model,
        model_name=model_name,
        modules=["v_proj"],
    )["v_proj"]


def get_q_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
) -> torch.nn.Linear:
    return _get_qkv_proj_module(
        architecture=architecture,
        layer_index=layer_index,
        model=model,
        model_name=model_name,
        modules=["q_proj"],
    )["q_proj"]


def get_k_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
) -> torch.nn.Linear:
    return _get_qkv_proj_module(
        architecture=architecture,
        layer_index=layer_index,
        model=model,
        model_name=model_name,
        modules=["k_proj"],
    )["k_proj"]


def get_rope_module(
    architecture: BaseModelArchitecture,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
) -> torch.nn.Module:

    if model is None:
        load_model_inside_function = True
    else:
        load_model_inside_function = False

    if load_model_inside_function:
        assert model_name is not None, "model_name must be provided if model is None"
        model = load_causal_model(model_name)

    model_module = getattr(model, architecture.model_field)
    assert architecture.rope_field is not None, (
        "Architecture does not specify a rope field, but attn_use_rope is True."
    )
    rope_module = copy.deepcopy(getattr(model_module, architecture.rope_field))
    if load_model_inside_function:
        del model_module
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return rope_module


def get_rope_frequencies(
    architecture: BaseModelArchitecture,
    model_config: PreTrainedConfig,
    layer_index: int,
    rope_module: torch.nn.Module | None = None,
    model: PreTrainedModel | None = None,
    model_name: str | None = None,
) -> tuple[torch.Tensor, float]:
    """Return (inv_freq, attention_scaling) for the given layer.

    For most architectures `model.rotary_emb` holds a single global
    inv_freq/attention_scaling pair (architecture.config_layer_types is None).

    Gemma3 instead keeps separate buffers per layer type (e.g.
    `sliding_attention_inv_freq` vs `full_attention_inv_freq`, since sliding
    and full-attention layers use different rope_theta), selected via
    `model_config.layer_types[layer_index]`.
    """
    if rope_module is None:
        rope_module = get_rope_module(
            architecture=architecture, model=model, model_name=model_name
        )

    if architecture.config_layer_types is None:
        return rope_module.inv_freq, rope_module.attention_scaling

    layer_type = getattr(model_config, architecture.config_layer_types)[layer_index]
    return (
        getattr(rope_module, f"{layer_type}_inv_freq"),
        getattr(rope_module, f"{layer_type}_attention_scaling"),
    )
