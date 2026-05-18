import pytest
import torch
from transformers import AutoModelForCausalLM

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.extractor.extractor_test import _create_feature_config
from feature_extractor.hooks.embedding import EmbeddingHookManager
from feature_extractor.hooks.layer import LayerHookManager
from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
    load_tokenizer,
)
from feature_extractor.models.get_config import get_num_layers


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_hook(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    architecture = get_model_architecture(model)
    feature_config = _create_feature_config()

    hook_manager = LayerHookManager(
        model=model, architecture=architecture, feature_cfg=feature_config
    )
    assert hook_manager.output_layer_indices == [0], (
        f"Expected output_layer_indices [0], got {hook_manager.output_layer_indices}"
    )
    assert hook_manager.input_layer_indices == [0], (
        f"Expected input_layer_indices [0], got {hook_manager.input_layer_indices}"
    )
    assert hook_manager.layer_indices == [0], (
        f"Expected layer_indices [0], got {hook_manager.layer_indices}"
    )

    assert len(hook_manager.layer_hooks) == 1, (
        f"Expected 1 combined hook, got {len(hook_manager.layer_hooks)}"
    )

    tokenizer = load_tokenizer(model_name)

    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        output = model(
            **inputs, return_dict_in_generate=True, output_hidden_states=True
        )

    assert isinstance(
        hook_manager.layer_hooks[0].result.hidden_states.shape, torch.Size
    )

    assert torch.allclose(
        hook_manager.layer_hooks[0].result.hidden_states_output,
        output["hidden_states"][1][0],
    )

    hook_manager.remove_hooks()


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_layer_input_equals_embedding_output_rope(model_name):
    """Layer 0 input should equal the embedding output on RoPE models."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    architecture = get_model_architecture(model)

    if not architecture.attn_use_rope:
        pytest.skip(f"Model {model_name} does not use RoPE, skipping")

    feature_config = FeatureConfig.from_str(
        feature_names=[
            "embeddings",
            "layers.layer_00.input",
        ],
        batch_size=16,
    )

    embedding_hook_manager = EmbeddingHookManager(
        model=model, architecture=architecture, feature_cfg=feature_config
    )
    layer_hook_manager = LayerHookManager(
        model=model, architecture=architecture, feature_cfg=feature_config
    )

    tokenizer = load_tokenizer(model_name)
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    num_layers = get_num_layers(model.config, architecture)
    embedding_output = embedding_hook_manager.get_features().output
    layer_result = layer_hook_manager.get_features(num_layers=num_layers)[0]

    assert layer_result is not None
    layer_input = layer_result.input

    assert embedding_output is not None
    assert layer_input is not None
    # Compare the first batch element
    assert torch.allclose(embedding_output[0], layer_input[0]), (
        "Layer 0 input should equal embedding output on RoPE models"
    )

    embedding_hook_manager.remove_hooks()
    layer_hook_manager.remove_hooks()
