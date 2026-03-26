import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS


def _create_feature_config():
    return FeatureConfig(
        feature_names=[
            "embeddings",
            "layers.layer_00.output",
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
def test_base_feature_extractor_initialization(model_name):
    config = _create_feature_config()
    extractor = FeatureExtractor(
        model_name_or_path=model_name, feature_cfg=config, hook_dtype=torch.float16
    )
    assert extractor.model is not None
    assert extractor.tokenizer is not None


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_base_feature_extractor(model_name):
    config = _create_feature_config()
    extractor = FeatureExtractor(
        model_name_or_path=model_name, feature_cfg=config, hook_dtype=torch.float16
    )
    assert extractor.model is not None
    assert extractor.tokenizer is not None

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )
    for batch, hook_result in extractor.extract_features(dataloader):
        assert batch["indices"] == ["0", "1"]
        assert hook_result.layers[0] is not None
        assert (
            len(hook_result.layers[0].output.shape) == 3
        )  # (batch_size, seq_len, hidden_dim)
        assert hook_result.layers[0].output.shape[0] == 2
        assert hook_result.layers[1] is None
