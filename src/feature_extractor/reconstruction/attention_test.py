import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS
from feature_extractor.models.llama import LlamaArchitecture
from feature_extractor.reconstruction import reconstruct_attention_weights


def _create_feature_config():
    return FeatureConfig(
        feature_names=[
            "attn.layer_00.query",
            "attn.layer_00.key",
            "attn.layer_00.attn_weights",
            "attn.layer_00.attention_mask",
            "attn.layer_00.positional_embedding",
        ],
        output_dir="outputs/test_reconstruction",
        save_format="pt",
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
def test_attention_reconstruction_accuracy(model_name):
    config = _create_feature_config()
    extractor = FeatureExtractor(model_name_or_path=model_name, feature_cfg=config)
    is_rope_model = "llama" in model_name.lower()
    assert (
        extractor.architecture.attn_position_embeddings_arg_name is not None
    ) == is_rope_model

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )

    for batch, hook_result in extractor.extract_features(dataloader):
        assert "input_ids" in batch
        attn_result = hook_result.attn[0]
        assert attn_result is not None
        assert attn_result.query is not None
        assert attn_result.key is not None
        assert attn_result.attn_weights is not None

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
