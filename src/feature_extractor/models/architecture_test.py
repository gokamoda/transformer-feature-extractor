from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
)


def test_supported_models_have_architectures():
    for model_name in SUPPORTED_MODELS:
        architecture = get_model_architecture(model_name)
        assert architecture is not None, (
            f"Model {model_name} does not have a defined architecture"
        )
