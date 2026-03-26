import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TokenizersBackend,
)

from utils.logger import init_logging

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
