import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS
from feature_extractor.models.get_modules import get_o_proj_module
from feature_extractor.reconstruction.attention_output import (
    reconstruct_attn_output,
)


def _create_feature_config():
    return FeatureConfig.from_str(
        feature_names=[
            "layers.layer_00.output",
            "attn.layer_01.query",
            "attn.layer_01.key",
            "attn.layer_01.value",
            "attn.layer_01.attn_weights",
            "attn.layer_01.attention_mask",
            "attn.layer_01.positional_embedding",
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
    attn_result = hook_result.attn[1]
    assert attn_result is not None
    assert attn_result.query is not None
    assert attn_result.key is not None
    assert attn_result.attn_weights is not None
    if extractor.architecture.attn_position_embeddings_arg_name is not None:
        assert attn_result.position_embeddings is not None
    else:
        assert attn_result.position_embeddings is None

    o_proj_module = get_o_proj_module(
        model=extractor.model,
        architecture=extractor.architecture,
        layer_index=1,
    )
    reconstructed_output = reconstruct_attn_output(
        attn_weights=attn_result.attn_weights,
        value=attn_result.value,
        o_proj_module=o_proj_module,
        unfurl=unfurl,
    ).detach().cpu()
    if unfurl == "head_wise":
        reconstructed_output = reconstructed_output.sum(dim=-2)
        if o_proj_module.bias is not None:
            reconstructed_output = reconstructed_output + o_proj_module.bias.to(reconstructed_output.device)
        torch.testing.assert_close(
            reconstructed_output, attn_result.output, atol=1e-2, rtol=1e-2
        )
    elif unfurl == "token_wise":
        reconstructed_output = reconstructed_output.sum(dim=-2).sum(dim=-2)
        if o_proj_module.bias is not None:
            reconstructed_output = reconstructed_output + o_proj_module.bias.to(reconstructed_output.device)
        torch.testing.assert_close(
            reconstructed_output, attn_result.output, atol=2e-2, rtol=2e-2
        )
    else:
        torch.testing.assert_close(reconstructed_output, attn_result.output)
