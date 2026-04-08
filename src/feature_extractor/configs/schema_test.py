from feature_extractor.configs.schema import FeatureConfig


def test_config_schema():

    FeatureConfig(
        feature_names=[
            "embeddings",
            "layers.layer_00.input",
            "layers.layer_00.output",
            "layers.layer_11.output",
        ],
        output_dir="outputs/test_features",
        save_format="pt",
        batch_size=16,
    )

    assert True
