from typing import Any, cast

import pytest

from feature_extractor.models import SUPPORTED_MODELS, load_causal_model, load_tokenizer


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_load_models(model_name):
    model = load_causal_model(model_name)
    assert model is not None


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_load_tokenizers(model_name):
    tokenizer = load_tokenizer(model_name)
    assert tokenizer is not None


def test_gpt2_gets_bos_and_left_padding():
    tokenizer = load_tokenizer("openai-community/gpt2")
    ids = tokenizer("Tokyo")["input_ids"]
    assert tokenizer.convert_ids_to_tokens(ids)[0] == "<|endoftext|>"
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id


def test_other_tokenizers_keep_their_tokens():
    from transformers import AutoTokenizer

    name = "HuggingFaceTB/SmolLM2-135M"
    plain = cast(Any, AutoTokenizer.from_pretrained(name))
    fixed = load_tokenizer(name)
    assert fixed("Tokyo")["input_ids"] == plain("Tokyo")["input_ids"]
