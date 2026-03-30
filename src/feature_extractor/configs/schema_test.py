import pytest

from feature_extractor.configs.schema import FeatureConfig


def test_feature_config_accepts_attn_position_embeddings_and_attention_mask():
    config = FeatureConfig(
        feature_names=[
            "attn.layer_00.position_embeddings",
            "attn.layer_00.attention_mask",
        ]
    )
    assert config.feature_names == [
        "attn.layer_00.position_embeddings",
        "attn.layer_00.attention_mask",
    ]


def test_feature_config_rejects_unknown_attention_feature_name():
    with pytest.raises(ValueError):
        FeatureConfig(feature_names=["attn.layer_00.unknown_feature"])
