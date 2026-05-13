import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS
from feature_extractor.models.llama import LlamaArchitecture
from feature_extractor.reconstruction.attention import (
    reconstruct_attention_weights,
    reconstruct_attn_output,
)


def _create_feature_config():
    return FeatureConfig.from_str(
        feature_names=[
            "attn.layer_00.query",
            "attn.layer_00.key",
            "attn.layer_00.value",
            "attn.layer_00.attn_weights",
            "attn.layer_00.attention_mask",
            "attn.layer_00.positional_embedding",
            "attn.layer_00.output",
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
def test_attention_weight_reconstruction_accuracy(model_name):
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
    assert "input_ids" in batch
    attn_result = hook_result.attn[0]
    assert attn_result is not None
    assert attn_result.query is not None
    assert attn_result.key is not None
    assert attn_result.attn_weights is not None
    if extractor.architecture.attn_position_embeddings_arg_name is not None:
        assert attn_result.position_embeddings is not None
    else:
        assert attn_result.position_embeddings is None

    reconstructed = reconstruct_attention_weights(
        query=attn_result.query,
        key=attn_result.key,
        attention_mask=attn_result.attention_mask,
        position_embeddings=attn_result.position_embeddings,
        architecture=extractor.architecture,
    )

    torch.testing.assert_close(
        reconstructed, attn_result.attn_weights, rtol=1e-4, atol=1e-4
    )


def test_attention_reconstruction_requires_rope_embeddings():
    architecture = LlamaArchitecture()
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 2, 4)
    with pytest.raises(
        ValueError,
        match="RoPE-based architectures require position embeddings.",
    ):
        reconstruct_attention_weights(
            query=query,
            key=key,
            attention_mask=None,
            position_embeddings=None,
            architecture=architecture,
        )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
@pytest.mark.parametrize("unfurl", ["none", "head_wise", "token_wise"])
def test_attention_reconstruction_accuracy(model_name, unfurl):
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
    assert "input_ids" in batch
    attn_result = hook_result.attn[0]
    assert attn_result is not None
    assert attn_result.query is not None
    assert attn_result.key is not None
    assert attn_result.attn_weights is not None
    if extractor.architecture.attn_position_embeddings_arg_name is not None:
        assert attn_result.position_embeddings is not None
    else:
        assert attn_result.position_embeddings is None

    model_module = getattr(extractor.model, extractor.architecture.model_field)
    layer_module = getattr(model_module, extractor.architecture.layers_field)[0]
    attn_module = getattr(layer_module, extractor.architecture.attn_field)
    o_proj_module = getattr(attn_module, extractor.architecture.attn_o_proj_field)
    reconstructed_output = reconstruct_attn_output(
        attn_weights=attn_result.attn_weights,
        value=attn_result.value,
        o_proj_module=o_proj_module,
        unfurl=unfurl,
    )
    if unfurl == "head_wise":
        reconstructed_output = reconstructed_output.sum(dim=-2)
        if o_proj_module.bias is not None:
            reconstructed_output = reconstructed_output + o_proj_module.bias
        torch.testing.assert_close(
            reconstructed_output, attn_result.output, atol=1e-2, rtol=1e-2
        )
    elif unfurl == "token_wise":
        reconstructed_output = reconstructed_output.sum(dim=-2).sum(dim=-2)
        if o_proj_module.bias is not None:
            reconstructed_output = reconstructed_output + o_proj_module.bias
        torch.testing.assert_close(
            reconstructed_output, attn_result.output, atol=1e-2, rtol=1e-2
        )
    else:
        torch.testing.assert_close(reconstructed_output, attn_result.output)
