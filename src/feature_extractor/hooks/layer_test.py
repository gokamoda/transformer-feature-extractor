import pytest
import torch
from transformers import AutoModelForCausalLM

from feature_extractor.extractor.extractor_test import _create_feature_config
from feature_extractor.hooks.layer import LayerHookManager
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

    hook_manager = LayerHookManager(
        model=model, architecture=architecture, feature_cfg=feature_config
    )
    assert hook_manager.layer_indices == [0], (
        f"Expected layer_indices [0], got {hook_manager.layer_indices}"
    )

    assert len(hook_manager.layer_hooks) == 1, (
        f"Expected 1 hook, got {len(hook_manager.layer_hooks)}"
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
        hook_manager.layer_hooks[0].result.hidden_states, output["hidden_states"][1][0]
    )

    hook_manager.remove_hooks()
