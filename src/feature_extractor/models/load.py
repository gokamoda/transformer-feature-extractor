from __future__ import annotations

from importlib import import_module
from types import ModuleType

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utils.logger import init_logging

logger = init_logging(__name__)


def _load_torch_module() -> ModuleType:
    return import_module("torch")


def _model_load_kwargs(torch_module: ModuleType | None = None) -> dict[str, object]:
    torch_module = torch_module or _load_torch_module()
    if not torch_module.cuda.is_available():
        return {}
    if torch_module.cuda.device_count() > 1:
        return {"device_map": "auto"}
    return {}


def load_causal_model(model_name_or_path: str) -> PreTrainedModel:
    torch_module = _load_torch_module()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **_model_load_kwargs(torch_module),
    )
    if torch_module.cuda.is_available() and torch_module.cuda.device_count() == 1:
        model = model.to("cuda")

    model.eval()
    logger.info("Model loaded on device: %s", model.device)
    return model


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
