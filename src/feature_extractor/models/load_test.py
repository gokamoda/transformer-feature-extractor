from __future__ import annotations

from types import SimpleNamespace

import pytest

import feature_extractor.models.load as load_module
from feature_extractor.models import SUPPORTED_MODELS, load_causal_model, load_tokenizer


class _FakeCuda:
    def __init__(self, *, available: bool, count: int) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _FakeTorch:
    def __init__(self, *, available: bool, count: int) -> None:
        self.cuda = _FakeCuda(available=available, count=count)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_load_models(monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    captured: dict[str, object] = {}
    fake_model = SimpleNamespace(eval=lambda: None, device="cpu")

    def fake_from_pretrained(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(
        load_module,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=fake_from_pretrained),
    )
    monkeypatch.setattr(
        "feature_extractor.models.load._load_torch_module",
        lambda: _FakeTorch(available=False, count=0),
    )

    model = load_causal_model(model_name)

    assert model is fake_model
    assert captured == {"name": model_name, "kwargs": {}}


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_load_models_use_device_map_for_multi_gpu(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    captured: dict[str, object] = {}
    fake_model = SimpleNamespace(eval=lambda: None, device="cuda:0")

    def fake_from_pretrained(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(
        load_module,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=fake_from_pretrained),
    )
    monkeypatch.setattr(
        "feature_extractor.models.load._load_torch_module",
        lambda: _FakeTorch(available=True, count=2),
    )

    model = load_causal_model(model_name)

    assert model is fake_model
    assert captured == {"name": model_name, "kwargs": {"device_map": "auto"}}


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_load_tokenizers(monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    captured: dict[str, object] = {}
    fake_tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=99)

    def fake_from_pretrained(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return fake_tokenizer

    monkeypatch.setattr(
        load_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    tokenizer = load_tokenizer(model_name)

    assert tokenizer is fake_tokenizer
    assert tokenizer.pad_token_id == 99
    assert captured == {
        "name": model_name,
        "kwargs": {"padding_side": "left"},
    }
