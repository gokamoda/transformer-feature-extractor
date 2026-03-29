import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS
from feature_extractor.reconstruction import reconstruct_attention_weights


def _create_feature_config():
    return FeatureConfig(
        feature_names=[
            "attn.layer_00.query",
            "attn.layer_00.key",
            "attn.layer_00.attn_weights",
        ],
        output_dir="outputs/test_features",
        save_format="pt",
        batch_size=16,
    )


def _create_dataset():
    return TextDataset(
        data=[
            TextDataEntry(idx="0", text="Hello, world!"),
            TextDataEntry(idx="1", text="Testing feature reconstruction."),
        ]
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_reconstruct_attention_weights_matches_hook(model_name: str):
    config = _create_feature_config()
    extractor = FeatureExtractor(model_name_or_path=model_name, feature_cfg=config)

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )

    for batch, hook_result in extractor.extract_features(dataloader):
        assert hook_result.attn is not None
        assert hook_result.attn[0] is not None

        query = hook_result.attn[0].query
        key = hook_result.attn[0].key
        attn_weights = hook_result.attn[0].attn_weights

        assert query is not None
        assert key is not None
        assert attn_weights is not None

        position_ids = None
        if extractor.architecture.absolute_pos_embedding_field is None:
            position_ids = batch["attention_mask"].cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(batch["attention_mask"] == 0, 0)

        reconstructed = reconstruct_attention_weights(
            query=query,
            key=key,
            attention_mask=batch["attention_mask"],
            position_ids=position_ids,
            model=extractor.model,
            architecture=extractor.architecture,
            layer_index=0,
        )

        torch.testing.assert_close(
            reconstructed,
            attn_weights,
            rtol=1e-4,
            atol=1e-4,
        )
        break
