import pytest

from feature_extractor.models import SUPPORTED_MODELS, load_causal_model, load_tokenizer


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_load_models(model_name):
    model = load_causal_model(model_name)
    assert model is not None


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS))
def test_load_tokenizers(model_name):
    tokenizer = load_tokenizer(model_name)
    assert tokenizer is not None
