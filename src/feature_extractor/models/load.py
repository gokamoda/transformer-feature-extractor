from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from utils.logger import init_logging

logger = init_logging(__name__)


def _model_load_kwargs() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {}
    if torch.cuda.device_count() > 1:
        return {"device_map": "auto"}
    return {}


def load_causal_model(model_name_or_path: str) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **_model_load_kwargs(),
    )
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.to("cuda")

    model.eval()
    logger.info("Model loaded on device: %s", model.device)
    return model


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
