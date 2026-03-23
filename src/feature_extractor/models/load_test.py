from __future__ import annotations

from types import SimpleNamespace

import pytest

from feature_extractor.models import SUPPORTED_MODELS, load_causal_model, load_tokenizer


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_load_models(monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    captured: dict[str, object] = {}
    fake_model = SimpleNamespace(eval=lambda: None, device="cpu")

    def fake_from_pretrained(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(
        "feature_extractor.models.load.AutoModelForCausalLM.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "feature_extractor.models.load.torch.cuda.is_available", lambda: False
    )

    model = load_causal_model(model_name)

    assert model is fake_model
    assert captured == {"name": model_name, "kwargs": {}}


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_load_tokenizers(monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    captured: dict[str, object] = {}
    fake_tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=99)

    def fake_from_pretrained(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return fake_tokenizer

    monkeypatch.setattr(
        "feature_extractor.models.load.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    tokenizer = load_tokenizer(model_name)

    assert tokenizer is fake_tokenizer
    assert tokenizer.pad_token_id == 99
    assert captured == {
        "name": model_name,
        "kwargs": {"padding_side": "left"},
    }
