from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

MLP_IMPLEMENTATION_STANDARD = "standard"
MLP_IMPLEMENTATION_GATED = "gated"
QKV_IMPLEMENTATION_INDEPENDENT_LINEAR = "independent_linear"
QKV_IMPLEMENTATION_CONV1D = "conv1d"

ArchitectureFactory = Callable[[], "BaseModelArchitecture"]


@dataclass(frozen=True)
class HookRegistrationConfig:
    """Model-specific forward-hook registration settings."""

    with_kwargs: bool = False
    input_arg_names: tuple[str, ...] = ()
    input_kwarg_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaseModelArchitecture:
    """Describe model attribute names for feature extraction hooks."""

    model_field: str = "model"
    layer_field: str = "layer"
    attn_field: str = "attn"
    mlp_field: str = "mlp"
    qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
    )
    mlp_implementation: Literal["standard", "gated"] = MLP_IMPLEMENTATION_STANDARD
    attention_hook_config: HookRegistrationConfig = field(
        default_factory=HookRegistrationConfig
    )
    mlp_hook_config: HookRegistrationConfig = field(default_factory=HookRegistrationConfig)


@dataclass(frozen=True)
class LlamaForCausalLMArchitecture(BaseModelArchitecture):
    model_field: str = "model"
    layer_field: str = "layers"
    attn_field: str = "self_attn"


@dataclass(frozen=True)
class GPT2LMHeadModelArchitecture(BaseModelArchitecture):
    model_field: str = "transformer"
    layer_field: str = "h"
    attn_field: str = "attn"
    qkv_implementation: Literal["conv1d", "independent_linear"] = (
        QKV_IMPLEMENTATION_CONV1D
    )


_ARCHITECTURE_REGISTRY: dict[str, ArchitectureFactory] = {
    "GPT2LMHeadModel": GPT2LMHeadModelArchitecture,
    "LlamaForCausalLM": LlamaForCausalLMArchitecture,
    "MistralForCausalLM": LlamaForCausalLMArchitecture,
}

_ARCHITECTURE_PREFIX_REGISTRY: dict[str, ArchitectureFactory] = {
    "Llama": LlamaForCausalLMArchitecture,
    "Mistral": LlamaForCausalLMArchitecture,
}


def get_model_architecture(architecture_name: str) -> BaseModelArchitecture:
    factory = _ARCHITECTURE_REGISTRY.get(architecture_name)
    if factory is not None:
        return factory()

    for prefix, prefix_factory in _ARCHITECTURE_PREFIX_REGISTRY.items():
        if architecture_name.startswith(prefix):
            return prefix_factory()

    msg = (
        f"Unsupported architecture: {architecture_name}. "
        f"Known architectures: {sorted(_ARCHITECTURE_REGISTRY)}"
    )
    raise ValueError(msg)
