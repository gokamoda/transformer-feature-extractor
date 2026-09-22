from dataclasses import dataclass
from typing import Callable

from transformers import (
    AutoConfig,
    PreTrainedModel,
)

from feature_extractor.logger import init_logging

from .architecture import BaseModelArchitecture
from .get_config import (
    get_hidden_size,
    get_hidden_size_per_head,
    get_intermediate_size,
    get_kv_hidden_size,
    get_num_attn_heads,
    get_num_kv_heads,
    get_num_layers,
)
from .get_modules import get_o_proj_module, get_pre_attn_norm_module, get_v_proj_module
from .gemma3 import Gemma3Architecture
from .gpt2 import GPT2Architecture
from .llama import LlamaArchitecture
from .load import load_causal_model, load_tokenizer
from .qwen3 import Qwen3Architecture

SUPPORTED_MODELS = [
    "openai-community/gpt2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.2-1B",
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B",
    "google/gemma-3-1b-pt",
]

__all__ = [
    "load_causal_model",
    "load_tokenizer",
    "SUPPORTED_MODELS",
    "get_model_architecture",
    "get_num_layers",
    "BaseModelArchitecture",
    "resolve_model_architecture",
    "get_num_attn_heads",
    "get_num_kv_heads",
    "get_hidden_size",
    "get_hidden_size_per_head",
    "get_kv_hidden_size",
    "get_intermediate_size",
    "get_pre_attn_norm_module",
    "get_v_proj_module",
    "get_o_proj_module",
]


logger = init_logging(__name__)


@dataclass(frozen=True)
class ArchitectureRegistryEntry:
    matcher: Callable[[str], bool]
    factory: Callable[[], BaseModelArchitecture]


def callable_name(fn: Callable[..., object]) -> str:
    return getattr(fn, "__name__", fn.__class__.__name__)


ARCHITECTURE_REGISTRY: tuple[ArchitectureRegistryEntry, ...] = (
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "LlamaForCausalLM" in class_name,
        factory=LlamaArchitecture,
    ),
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "GPT2LMHeadModel" in class_name,
        factory=GPT2Architecture,
    ),
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "Gemma3ForCausalLM" in class_name,
        factory=Gemma3Architecture,
    ),
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "Qwen3ForCausalLM" in class_name,
        factory=Qwen3Architecture,
    ),
    # Qwen2.5 and Mistral match BaseModelArchitecture's defaults exactly
    # (self_attn has only q_proj/k_proj/v_proj/o_proj, mlp has only
    # gate_proj/up_proj/down_proj/act_fn, a single model.rotary_emb), so they
    # reuse LlamaArchitecture rather than defining a new class.
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "Qwen2ForCausalLM" in class_name,
        factory=LlamaArchitecture,
    ),
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "MistralForCausalLM" in class_name,
        factory=LlamaArchitecture,
    ),
)


def resolve_model_architecture(model_class_name: str) -> BaseModelArchitecture:
    for entry in ARCHITECTURE_REGISTRY:
        if entry.matcher(model_class_name):
            print(
                f"Matched model class {model_class_name} to architecture {callable_name(entry.factory)}"
            )
            return entry.factory()

    logger.warning(
        f"Model class name {model_class_name} not recognized. Using default architecture."
    )
    return BaseModelArchitecture()


def get_model_architecture(
    model: PreTrainedModel | type[PreTrainedModel] | str,
) -> BaseModelArchitecture:
    """
    Return a BaseModelArchitecture with appropriate field names for the given model.

    Accepts a model instance, model class, or class name string.
    """
    if isinstance(model, str):
        model_class_name = AutoConfig.from_pretrained(model).architectures[0]
    elif isinstance(model, type):
        model_class_name = model.__name__
    else:
        model_class_name = model.__class__.__name__

    return resolve_model_architecture(model_class_name)
