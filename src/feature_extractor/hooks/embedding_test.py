import pytest
import torch
from transformers import AutoModelForCausalLM

from feature_extractor.extractor.extractor_test import _create_feature_config
from feature_extractor.hooks.embedding import EmbeddingHookManager
from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
    load_tokenizer,
)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_hook(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    architecture = get_model_architecture(model)
    feature_config = _create_feature_config()

    hook_manager = EmbeddingHookManager(
        model=model, architecture=architecture, feature_cfg=feature_config
    )
    assert hook_manager.embedding_hook is not None

    tokenizer = load_tokenizer(model_name)
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    embedding_input = inputs["input_ids"]

    with torch.no_grad():
        model(**inputs)

    embedding_module = getattr(
        getattr(model, architecture.model_field),
        architecture.word_embedding_field,
    )
    expected_embedding = embedding_module(embedding_input)

    assert isinstance(
        hook_manager.embedding_hook.result.embeddings.shape, torch.Size
    )
    assert torch.allclose(
        hook_manager.embedding_hook.result.embeddings,
        expected_embedding,
    )

    hook_manager.remove_hooks()
