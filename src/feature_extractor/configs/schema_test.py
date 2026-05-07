from feature_extractor.configs.schema import (
    AttentionFeatureSpec,
    EmbeddingFeatureSpec,
    FeatureConfig,
    LayerFeatureSpec,
    MLPFeatureSpec,
)


def test_config_schema():

    FeatureConfig(
        feature_specs=[
            EmbeddingFeatureSpec(),
            LayerFeatureSpec(layer_index=0, feature="input"),
            LayerFeatureSpec(layer_index=0, feature="output"),
            LayerFeatureSpec(layer_index=11, feature="output"),
            AttentionFeatureSpec(layer_index=2, feature="query"),
            MLPFeatureSpec(layer_index=3, feature="output"),
        ],
        batch_size=16,
    )

    assert True


def test_config_schema_accepts_legacy_strings():
    cfg = FeatureConfig.from_str(
        feature_names=[
            "embeddings",
            "layers.layer_00.input",
            "attn.layer_02.query",
            "mlp.layer_03.output",
        ],
        batch_size=16,
    )

    assert isinstance(cfg.feature_specs[0], EmbeddingFeatureSpec)
    assert isinstance(cfg.feature_specs[1], LayerFeatureSpec)
    assert isinstance(cfg.feature_specs[2], AttentionFeatureSpec)
    assert isinstance(cfg.feature_specs[3], MLPFeatureSpec)
