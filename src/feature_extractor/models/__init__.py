from dataclasses import dataclass
from typing import Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from feature_extractor.logger import init_logging

from .base_architecture import BaseModelArchitecture
from .gpt2 import GPT2Architecture
from .llama import LlamaArchitecture

SUPPORTED_MODELS = [
    "openai-community/gpt2",
    # "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.2-1B",
]

__all__ = [
    "load_causal_model",
    "load_tokenizer",
    "SUPPORTED_MODELS",
    "get_model_architecture",
    "BaseModelArchitecture",
    "resolve_model_architecture",
]


logger = init_logging(__name__)


@dataclass(frozen=True)
class ArchitectureRegistryEntry:
    matcher: Callable[[str], bool]
    factory: Callable[[], BaseModelArchitecture]


ARCHITECTURE_REGISTRY: tuple[ArchitectureRegistryEntry, ...] = (
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "LlamaForCausalLM" in class_name,
        factory=LlamaArchitecture,
    ),
    ArchitectureRegistryEntry(
        matcher=lambda class_name: "GPT2LMHeadModel" in class_name,
        factory=GPT2Architecture,
    ),
)


def load_causal_model(model_name_or_path: str) -> PreTrainedModel:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            model = model.to("cuda")  # ty: ignore
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    model.eval()
    logger.info(f"Model loaded on device: {model.device}")
    return model


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def resolve_model_architecture(model_class_name: str) -> BaseModelArchitecture:
    for entry in ARCHITECTURE_REGISTRY:
        if entry.matcher(model_class_name):
            return entry.factory()

    logger.warning(
        f"Model class name {model_class_name} not recognized. Using default architecture."
    )
    return BaseModelArchitecture()


def get_model_architecture(model: PreTrainedModel | type[PreTrainedModel] | str) -> BaseModelArchitecture:
    """
    Return a BaseModelArchitecture with appropriate field names for the given model.

    Accepts a model instance, model class, or class name string.
    """
    if isinstance(model, str):
        model_class_name = model
    elif isinstance(model, type):
        model_class_name = model.__name__
    else:
        model_class_name = model.__class__.__name__

    return resolve_model_architecture(model_class_name)
