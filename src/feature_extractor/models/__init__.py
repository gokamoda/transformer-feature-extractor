import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TokenizersBackend,
)

from utils.logger import init_logging

from .base_architecture import BaseModelArchitecture
from .llama import LlamaArchitecture
from .gpt2 import GPT2Architecture

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
]


logger = init_logging(__name__)


def load_causal_model(model_name_or_path: str) -> PreTrainedModel:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto"
            )
        else:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name_or_path
            )
            model = model.to("cuda")  # ty: ignore
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    model.eval()
    logger.info(f"Model loaded on device: {model.device}")
    return model


def load_tokenizer(model_name_or_path: str) -> TokenizersBackend:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    assert isinstance(tokenizer, TokenizersBackend), tokenizer.__class__
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_model_architecture(model_class_name: str) -> BaseModelArchitecture:
    """
    Return a BaseModelArchitecture with appropriate field names for the given model class name.
    """
    if "LlamaForCausalLM" in model_class_name:
        return LlamaArchitecture()
    elif "GPT2LMHeadModel" in model_class_name:
        return GPT2Architecture()
    else:
        logger.warning(
            f"Model class name {model_class_name} not recognized. Using default architecture."
        )
        return BaseModelArchitecture()
