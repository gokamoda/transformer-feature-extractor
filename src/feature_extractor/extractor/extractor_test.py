import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS
from feature_extractor.models.architecture import (
    get_hidden_size,
    get_hidden_size_per_head,
    get_num_attn_heads,
)


def _create_feature_config():
    return FeatureConfig(
        feature_names=[
            "layers.layer_00.output",
            "attn.layer_00.query",
            "attn.layer_01.key",
            "attn.layer_00.value",
            "attn.layer_00.attn_weights",
            "attn.layer_01.output",
            "attn.layer_00.attention_mask",
            "attn.layer_00.positional_embedding",
            "mlp.layer_00.activation",
            "mlp.layer_01.output",
        ],
        output_dir="outputs/test_features",
        save_format="pt",
        batch_size=16,
    )


def _create_dataset():
    return TextDataset(
        data=[
            TextDataEntry(idx="0", text="Hello, world!"),
            TextDataEntry(idx="1", text="Testing feature extraction."),
        ]
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_feature_extractor_initialization(model_name):
    config = _create_feature_config()
    extractor = FeatureExtractor(
        model_name_or_path=model_name, feature_cfg=config, hook_dtype=torch.float16
    )
    assert extractor.model is not None
    assert extractor.tokenizer is not None


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_feature_extractor(model_name):
    config = _create_feature_config()
    extractor = FeatureExtractor(
        model_name_or_path=model_name, feature_cfg=config, hook_dtype=torch.float16
    )
    assert extractor.model is not None
    assert extractor.tokenizer is not None
    assert extractor.attn_hook is not None
    assert extractor.attn_hook.qkv_hook_manager is not None
    assert extractor.attn_hook.attn_module_hook_manager is not None
    assert extractor.mlp_hook is not None
    assert extractor.attn_hook.attn_module_hook_manager.layer_indices == [0, 1]
    assert extractor.attn_hook.attn_module_hook_manager.attn_weights_layer_indices == [
        0
    ]
    assert extractor.attn_hook.attn_module_hook_manager.output_layer_indices == [1]
    assert (
        extractor.attn_hook.attn_module_hook_manager.attention_mask_layer_indices == [0]
    )
    assert (
        extractor.attn_hook.attn_module_hook_manager.position_embeddings_layer_indices
        == [0]
    )
    assert len(extractor.attn_hook.attn_module_hook_manager.attn_module_hooks) == 2
    assert extractor.mlp_hook.activation_output_combined_layer_indices == [0, 1]
    assert extractor.mlp_hook.activation_layer_indices == [0]
    assert extractor.mlp_hook.output_layer_indices == [1]
    assert len(extractor.mlp_hook.activation_hooks) == 1
    assert len(extractor.mlp_hook.output_hooks) == 1

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )

    hidden_size = get_hidden_size(extractor.model.config, extractor.architecture)
    head_size = get_hidden_size_per_head(extractor.model.config, extractor.architecture)
    num_attn_heads = get_num_attn_heads(extractor.model.config, extractor.architecture)
    for batch, hook_result in extractor.extract_features(dataloader):
        # batch data
        assert batch["indices"] == ["0", "1"]

        # layer features
        assert hook_result.layers[0] is not None
        assert (
            len(hook_result.layers[0].output.shape) == 3
        )  # (batch_size, seq_len, hidden_dim)
        assert hook_result.layers[0].output.shape[0] == 2
        assert hook_result.layers[0].output.shape[2] == hidden_size
        assert hook_result.layers[1] is None

        # attention features
        assert hook_result.attn[0] is not None
        assert hook_result.attn[1] is not None

        assert hook_result.attn[0].query is not None
        assert hook_result.attn[1].query is None
        assert (
            len(hook_result.attn[0].query.shape) == 4
        )  # (batch_size, num_heads, seq_len, head_dim)
        assert hook_result.attn[0].query.shape[0] == 2
        assert hook_result.attn[0].query.shape[1] == num_attn_heads
        assert hook_result.attn[0].query.shape[-1] == head_size

        assert hook_result.attn[0].key is None
        assert hook_result.attn[1].key is not None
        assert (
            len(hook_result.attn[1].key.shape) == 4
        )  # (batch_size, num_kv_heads, seq_len, head_dim)
        assert hook_result.attn[1].key.shape[0] == 2
        assert hook_result.attn[1].key.shape[-1] == head_size

        assert hook_result.attn[0].value is not None
        assert hook_result.attn[1].value is None
        assert (
            len(hook_result.attn[0].value.shape) == 4
        )  # (batch_size, num_kv_heads, seq_len, head_dim)
        assert hook_result.attn[0].value.shape[0] == 2
        assert hook_result.attn[0].value.shape[-1] == head_size

        assert (
            hook_result.attn[0].query.shape[2]
            == hook_result.attn[1].key.shape[2]
            == hook_result.attn[0].value.shape[2]
        )  # seq_len

        # attention weights
        assert hook_result.attn[0].attn_weights is not None
        assert (
            len(hook_result.attn[0].attn_weights.shape) == 4
        )  # (batch_size, num_heads, seq_len, seq_len)
        assert hook_result.attn[0].attn_weights.shape[0] == 2
        assert hook_result.attn[0].attn_weights.shape[1] == num_attn_heads
        assert (
            hook_result.attn[0].attn_weights.shape[2]
            == hook_result.attn[0].attn_weights.shape[3]
        )  # seq_len

        # attention module inputs
        assert hook_result.attn[0].attention_mask is not None
        assert hook_result.attn[1].attention_mask is None
        assert hook_result.attn[0].attention_mask.shape[0] == 2

        if extractor.architecture.attn_position_embeddings_arg_name is not None:
            assert hook_result.attn[0].position_embeddings is not None
            assert hook_result.attn[1].position_embeddings is None

        # attention output
        assert hook_result.attn[0].output is None
        assert hook_result.attn[1].output is not None
        assert (
            len(hook_result.attn[1].output.shape) == 3
        )  # (batch_size, seq_len, hidden_dim)
        assert hook_result.attn[1].output.shape[0] == 2
        assert hook_result.attn[1].output.shape[2] == hidden_size

        # mlp output
        assert hook_result.mlp[0] is not None
        assert hook_result.mlp[1] is not None

        assert hook_result.mlp[0].activation is not None
        assert hook_result.mlp[1].activation is None
        assert len(hook_result.mlp[0].activation.shape) == 3
        assert hook_result.mlp[0].activation.shape[0] == 2

        assert hook_result.mlp[0].output is None
        assert hook_result.mlp[1].output is not None
        assert len(hook_result.mlp[1].output.shape) == 3
        assert hook_result.mlp[1].output.shape[0] == 2
        assert hook_result.mlp[1].output.shape[2] == hidden_size
