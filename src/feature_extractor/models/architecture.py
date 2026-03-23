from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BaseModelArchitecture:
    """Describe model attribute names for feature extraction hooks."""

    model_field: str = "model"
    layer_field: str = "layer"
    attn_field: str = "attn"
    mlp_field: str = "mlp"
    qkv_implementation: Literal["conv1d", "independent_linear"] = "independent_linear"
    mlp_implementation: Literal["standard", "gated"] = "standard"

class LlamaForCausalLMArchitecture(BaseModelArchitecture):
    model_field: str = "model"
    layer_field: str = "layers"
    attn_field: str = "self_attn"
    mlp_field: str = "mlp"
    qkv_implementation: Literal["conv1d", "independent_linear"] = "independent_linear"
    mlp_implementation: Literal["standard", "gated"] = "standard"


def get_model_architecture(architecture_name: str) -> BaseModelArchitecture:
    if architecture_name == "LlamaForCausalLM":
        return LlamaForCausalLMArchitecture()
    else:
        raise ValueError(f"Unsupported architecture: {architecture_name}")