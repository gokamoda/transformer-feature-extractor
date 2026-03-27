import pytest
import torch

from feature_extractor.extractor.extractor_test import _create_feature_config
from feature_extractor.hooks.attention import AttentionHookManager
from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
    load_causal_model,
    load_tokenizer,
)
from feature_extractor.models.architecture import (
    QKV_IMPLEMENTATION_CONV1D,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
    get_hidden_size,
    get_kv_hidden_size,
)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_attn_hook(model_name):
    model = load_causal_model(model_name)
    architecture = get_model_architecture(model)
    feature_config = _create_feature_config()

    hook_manager = AttentionHookManager(
        model=model, architecture=architecture, feature_cfg=feature_config
    )

    # resolve layer index
    assert hook_manager.query_layer_indices == [0], (
        f"Expected query_layer_indices [0], got {hook_manager.query_layer_indices}"
    )
    assert hook_manager.key_layer_indices == [1], (
        f"Expected key_layer_indices [1], got {hook_manager.key_layer_indices}"
    )
    assert hook_manager.value_layer_indices == [0], (
        f"Expected value_layer_indices [0], got {hook_manager.value_layer_indices}"
    )

    # install hooks
    if architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
        assert len(hook_manager.query_hooks) == 1, (
            f"Expected 1 hook, got {len(hook_manager.query_hooks)}"
        )
        assert len(hook_manager.key_hooks) == 1, (
            f"Expected 1 hook, got {len(hook_manager.key_hooks)}"
        )
        assert len(hook_manager.value_hooks) == 1, (
            f"Expected 1 hook, got {len(hook_manager.value_hooks)}"
        )
    elif architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
        assert hook_manager.qkv_combined_layer_indices == [0, 1], (
            f"Expected qkv_combined_layer_indices [0, 1], got {hook_manager.qkv_combined_layer_indices}"
        )
        assert len(hook_manager.qkv_combined_hooks) == 2, (
            f"Expected 2 hooks, got {len(hook_manager.qkv_combined_hooks)}"
        )

    tokenizer = load_tokenizer(model_name)

    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        model(**inputs, return_dict_in_generate=True, output_hidden_states=True)

    hidden_size = get_hidden_size(model.config, architecture)
    kv_size = get_kv_hidden_size(model.config, architecture)
    if architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
        assert isinstance(
            hook_manager.query_hooks[0].result.output.shape, torch.Size
        ), "Expected query hook result to have a tensor shape"
        assert hook_manager.query_hooks[0].result.output.shape[0] == 1, (
            f"Expected batch size 1 (batch size), got {hook_manager.query_hooks[0].result.output.shape[0]}"
        )
        assert hook_manager.query_hooks[0].result.output.shape[2] == hidden_size, (
            f"Expected hidden size {hidden_size} (hidden size), got {hook_manager.query_hooks[0].result.output.shape[2]}"
        )

        assert isinstance(hook_manager.key_hooks[0].result.output.shape, torch.Size), (
            "Expected key hook result to have a tensor shape"
        )
        assert hook_manager.key_hooks[0].result.output.shape[0] == 1, (
            f"Expected batch size 1 (batch size), got {hook_manager.key_hooks[0].result.output.shape[0]}"
        )
        assert hook_manager.key_hooks[0].result.output.shape[2] == kv_size, (
            f"Expected hidden size {kv_size} (hidden size), got {hook_manager.key_hooks[0].result.output.shape[2]}"
        )

        assert isinstance(
            hook_manager.value_hooks[0].result.output.shape, torch.Size
        ), "Expected value hook result to have a tensor shape"
        assert hook_manager.value_hooks[0].result.output.shape[0] == 1, (
            f"Expected batch size 1 (batch size), got {hook_manager.value_hooks[0].result.output.shape[0]}"
        )
        assert hook_manager.value_hooks[0].result.output.shape[2] == kv_size, (
            f"Expected hidden size {kv_size} (hidden size), got {hook_manager.value_hooks[0].result.output.shape[2]}"
        )

    elif architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
        for hook in hook_manager.qkv_combined_hooks:
            assert isinstance(hook.result.output.shape, torch.Size), (
                "Expected combined QKV hook result to have a tensor shape"
            )
            assert hook.result.output.shape[0] == 1, (
                f"Expected batch size 1 (batch size), got {hook.result.output.shape[0]}"
            )
            assert hook.result.output.shape[2] == hidden_size * 3, (
                f"Expected hidden size {hidden_size * 3} (hidden size * 3), got {hook.result.output.shape[2]}"
            )
