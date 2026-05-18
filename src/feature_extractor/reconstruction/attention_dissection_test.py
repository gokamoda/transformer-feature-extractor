import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS
from feature_extractor.models.get_config import get_num_attn_heads, get_num_kv_heads
from feature_extractor.models.get_modules import (
    _get_qkv_proj_module,
    get_o_proj_module,
    get_pre_attn_norm_module,
    get_rope_module,
    get_v_proj_module,
)
from feature_extractor.reconstruction.attention_dissection import (
    reconstruct_attn_output_vo_combined,
    reconstruct_attn_weight_qk_combined_norope,
    reconstruct_attn_weight_qk_combined_with_rope,
    reconstruct_qkv_vectors,
)
from feature_extractor.reconstruction.rope import SimplifiedRoPEV1
from feature_extractor.typing import Tensor


def _create_feature_config():
    return FeatureConfig.from_str(
        feature_names=[
            "layers.layer_00.output",
            "attn.layer_01.value",
            "attn.layer_01.query",
            "attn.layer_01.key",
            "attn.layer_01.attn_weights",
            "attn.layer_01.output",
        ],
        batch_size=16,
    )


def _create_dataset():
    return TextDataset(
        data=[
            TextDataEntry(idx="0", text="Hello, world!"),
            TextDataEntry(idx="1", text="Testing attention reconstruction."),
        ]
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_reconstruct_qkv_vectors(
    model_name,
):
    config = _create_feature_config()
    extractor = FeatureExtractor(model_name_or_path=model_name)
    extractor.configure(config)

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )

    batch, hook_result = next(extractor.extract_features(dataloader))
    hidden_states = hook_result.layers[0].output

    num_kv_heads = get_num_kv_heads(
        model_config=extractor.model.config, architecture=extractor.architecture
    )
    num_attention_heads = get_num_attn_heads(
        model_config=extractor.model.config, architecture=extractor.architecture
    )

    # layer normalize before attn
    ln_module = get_pre_attn_norm_module(
        architecture=extractor.architecture,
        model=extractor.model,
        layer_index=1,
    ).eval()
    with torch.no_grad():
        hidden_states = ln_module(hidden_states)

    # value
    value = hook_result.attn[1].value
    reconstructed_value = reconstruct_qkv_vectors(
        hidden_states=hidden_states,
        qkv_proj_module=_get_qkv_proj_module(
            model=extractor.model,
            architecture=extractor.architecture,
            layer_index=1,
            module="v_proj",
        ).eval(),
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_attention_heads,
        module_type="v_proj",
    )
    torch.testing.assert_close(reconstructed_value, value, atol=1e-4, rtol=1e-4)

    # query
    query = hook_result.attn[1].query
    reconstructed_query = reconstruct_qkv_vectors(
        hidden_states=hidden_states,
        qkv_proj_module=_get_qkv_proj_module(
            model=extractor.model,
            architecture=extractor.architecture,
            layer_index=1,
            module="q_proj",
        ).eval(),
        num_attention_heads=num_attention_heads,
        module_type="q_proj",
    )
    torch.testing.assert_close(reconstructed_query, query, atol=1e-4, rtol=1e-4)

    # key
    key = hook_result.attn[1].key
    reconstructed_key = reconstruct_qkv_vectors(
        hidden_states=hidden_states,
        qkv_proj_module=_get_qkv_proj_module(
            model=extractor.model,
            architecture=extractor.architecture,
            layer_index=1,
            module="k_proj",
        ).eval(),
        num_attention_heads=num_attention_heads,
        module_type="k_proj",
        num_kv_heads=num_kv_heads,
    )
    torch.testing.assert_close(reconstructed_key, key, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_attention_reconstruction_accuracy_ov_combined(model_name):
    config = _create_feature_config()
    extractor = FeatureExtractor(model_name_or_path=model_name)
    extractor.configure(config)

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )

    _, hook_result = next(extractor.extract_features(dataloader))
    attn_weights = hook_result.attn[1].attn_weights
    hidden_states = hook_result.layers[0].output
    attn_output = hook_result.attn[1].output

    # layer normalize before attn
    ln_module = get_pre_attn_norm_module(
        architecture=extractor.architecture,
        model=extractor.model,
        layer_index=1,
    ).eval()
    with torch.no_grad():
        hidden_states = ln_module(hidden_states)

    o_proj_module = get_o_proj_module(
        model=extractor.model,
        architecture=extractor.architecture,
        layer_index=1,
    ).eval()
    v_proj_module = get_v_proj_module(
        model=extractor.model,
        architecture=extractor.architecture,
        layer_index=1,
    ).eval()

    reconstructed_output = reconstruct_attn_output_vo_combined(
        attn_weights=attn_weights,
        hidden_states=hidden_states,
        v_proj_module=v_proj_module,
        o_proj_module=o_proj_module,
        num_attention_heads=get_num_attn_heads(
            model_config=extractor.model.config, architecture=extractor.architecture
        ),
        num_kv_heads=get_num_kv_heads(
            model_config=extractor.model.config, architecture=extractor.architecture
        ),
    )
    torch.testing.assert_close(reconstructed_output, attn_output, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_attention_weight_reconstruction_accuracy_qk_combined(model_name):
    extractor = FeatureExtractor(model_name_or_path=model_name)
    config = _create_feature_config()
    extractor.configure(config)

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )

    batch, hook_result = next(extractor.extract_features(dataloader))
    hidden_states = hook_result.layers[0].output

    num_kv_heads = get_num_kv_heads(
        model_config=extractor.model.config, architecture=extractor.architecture
    )
    num_attention_heads = get_num_attn_heads(
        model_config=extractor.model.config, architecture=extractor.architecture
    )

    q_proj_module = _get_qkv_proj_module(
        model=extractor.model,
        architecture=extractor.architecture,
        layer_index=1,
        module="q_proj",
    ).eval()
    k_proj_module = _get_qkv_proj_module(
        model=extractor.model,
        architecture=extractor.architecture,
        layer_index=1,
        module="k_proj",
    ).eval()
    head_dim = q_proj_module.out_features // num_attention_heads

    # layer normalize before attn
    ln_module = get_pre_attn_norm_module(
        architecture=extractor.architecture,
        model=extractor.model,
        layer_index=1,
    ).eval()
    with torch.no_grad():
        hidden_states = ln_module(hidden_states)

    if extractor.architecture.attn_use_rope:
        original_rope_module = get_rope_module(
            model=extractor.model,
            architecture=extractor.architecture,
        )
        attn_scaling = original_rope_module.attention_scaling
        if isinstance(attn_scaling, torch.Tensor):
            attn_scaling = float(attn_scaling.item())
        elif isinstance(attn_scaling, (int, float)):
            attn_scaling = float(attn_scaling)
        else:
            raise ValueError(
                f"Unexpected type for attention_scaling: {type(attn_scaling)}"
            )

        inv_freq = original_rope_module.inv_freq
        assert isinstance(inv_freq, Tensor)
        simplified_rope_module = SimplifiedRoPEV1(
            inv_freq=inv_freq,
            attention_scaling=attn_scaling,
        )
        attn_weights = reconstruct_attn_weight_qk_combined_with_rope(
            hidden_states=hidden_states,
            q_proj_module=q_proj_module,
            k_proj_module=k_proj_module,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            rope_module=simplified_rope_module,
        )
        torch.testing.assert_close(
            attn_weights, hook_result.attn[1].attn_weights, atol=1e-2, rtol=1e-2
        )
    else:
        attn_weights = reconstruct_attn_weight_qk_combined_norope(
            hidden_states=hidden_states,
            q_proj_module=q_proj_module,
            k_proj_module=k_proj_module,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )
        torch.testing.assert_close(
            attn_weights, hook_result.attn[1].attn_weights, atol=1e-4, rtol=1e-4
        )
