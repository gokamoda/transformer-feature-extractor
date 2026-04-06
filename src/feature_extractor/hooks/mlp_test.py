import pytest
import torch

from feature_extractor.extractor.extractor_test import _create_feature_config
from feature_extractor.hooks.mlp import MLPHookManager
from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
    load_causal_model,
    load_tokenizer,
)
from feature_extractor.models.architecture import get_hidden_size


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_mlp_hook(model_name):
    model = load_causal_model(model_name)
    architecture = get_model_architecture(model)
    feature_config = _create_feature_config()

    hook_manager = MLPHookManager(
        model=model,
        architecture=architecture,
        feature_cfg=feature_config,
    )

    assert hook_manager.activation_layer_indices == [0], (
        f"Expected activation_layer_indices [0], got {hook_manager.activation_layer_indices}"
    )
    assert hook_manager.down_proj_input_layer_indices == [0], (
        "Expected down_proj_input_layer_indices [0], got "
        f"{hook_manager.down_proj_input_layer_indices}"
    )
    assert hook_manager.output_layer_indices == [1], (
        f"Expected output_layer_indices [1], got {hook_manager.output_layer_indices}"
    )
    assert (
        hook_manager.activation_down_proj_input_output_combined_layer_indices == [0, 1]
    ), (
        "Expected activation_down_proj_input_output_combined_layer_indices [0, 1], got "
        f"{hook_manager.activation_down_proj_input_output_combined_layer_indices}"
    )

    assert len(hook_manager.activation_hooks) == 1, (
        f"Expected 1 activation hook, got {len(hook_manager.activation_hooks)}"
    )
    assert len(hook_manager.down_proj_input_hooks) == 1, (
        f"Expected 1 down projection input hook, got {len(hook_manager.down_proj_input_hooks)}"
    )
    assert len(hook_manager.output_hooks) == 1, (
        f"Expected 1 output hook, got {len(hook_manager.output_hooks)}"
    )

    tokenizer = load_tokenizer(model_name)

    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        model(
            **inputs,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    hidden_size = get_hidden_size(model.config, architecture)

    assert isinstance(
        hook_manager.activation_hooks[0].result.activation.shape,
        torch.Size,
    ), "Expected activation hook result to have a tensor shape"
    assert hook_manager.activation_hooks[0].result.activation.shape[0] == 1, (
        "Expected batch size 1 (batch size), got "
        f"{hook_manager.activation_hooks[0].result.activation.shape[0]}"
    )

    assert isinstance(
        hook_manager.output_hooks[0].result.output.shape,
        torch.Size,
    ), "Expected MLP output hook result to have a tensor shape"
    assert hook_manager.output_hooks[0].result.output.shape[0] == 1, (
        f"Expected batch size 1 (batch size), got {hook_manager.output_hooks[0].result.output.shape[0]}"
    )
    assert hook_manager.output_hooks[0].result.output.shape[2] == hidden_size, (
        f"Expected hidden size {hidden_size} (hidden size), got {hook_manager.output_hooks[0].result.output.shape[2]}"
    )

    assert isinstance(
        hook_manager.down_proj_input_hooks[0].result.down_proj_input.shape,
        torch.Size,
    ), "Expected down projection input hook result to have a tensor shape"
    assert hook_manager.down_proj_input_hooks[0].result.down_proj_input.shape[0] == 1, (
        "Expected batch size 1 (batch size), got "
        f"{hook_manager.down_proj_input_hooks[0].result.down_proj_input.shape[0]}"
    )
