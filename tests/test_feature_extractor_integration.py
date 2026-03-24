from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from feature_extractor.configs.schema import FeatureConfig  # noqa: E402
from feature_extractor.extractor.base import BaseFeatureExtractor  # noqa: E402
from feature_extractor.models import SUPPORTED_MODELS  # noqa: E402
from feature_extractor.models.architecture import get_model_architecture  # noqa: E402
from feature_extractor.reconstruction import reconstruct_attention_scores  # noqa: E402


def _patch_model_and_tokenizer(monkeypatch, model, tokenizer) -> None:
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )


def _final_norm_module_from_model(model_name: str, model: nn.Module) -> nn.Module:
    if model_name == "openai-community/gpt2":
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    msg = f"Could not resolve final norm module for model: {model_name}"
    raise ValueError(msg)


def _load_supported_model_or_skip(model_name: str):
    from feature_extractor.models.load import load_causal_model, load_tokenizer

    try:
        model = load_causal_model(model_name)
        tokenizer = load_tokenizer(model_name)
    except Exception as exc:  # pragma: no cover - env/network/auth dependent
        pytest.skip(f"Unable to load pretrained model {model_name}: {exc}")
    return model, tokenizer


def _resolve_rotary_module(model: nn.Module, architecture) -> nn.Module | None:
    model_root = getattr(model, architecture.model_field, model)
    rotary_module = getattr(model_root, "rotary_emb", None)
    if isinstance(rotary_module, nn.Module):
        return rotary_module
    first_layer = getattr(model_root, architecture.layer_field)[0]
    attn_module = getattr(first_layer, architecture.attn_field)
    nested_rotary = getattr(attn_module, "rotary_emb", None)
    if isinstance(nested_rotary, nn.Module):
        return nested_rotary
    return None


@pytest.mark.integration
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_supported_models_attention_and_final_norm_relationships_real_models(
    monkeypatch,
    model_name,
):
    model, tokenizer = _load_supported_model_or_skip(model_name)
    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(
        feature_names=[
            "embeddings",
            "layer.layer_00.attn_output",
            "layer.layer_00.output",
            "attn.layer_00.query",
            "attn.layer_00.key",
            "attn.layer_00.value",
            "attn.layer_00.weights",
        ],
        batch_size=1,
    )
    extractor = BaseFeatureExtractor("dummy", feature_cfg)

    sample_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer(
        [sample_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    dataset = [
        {
            "idx": "sample",
            "input_ids": encoded["input_ids"][0].detach().cpu(),
            "attention_mask": encoded["attention_mask"][0].detach().cpu(),
        }
    ]
    results = list(extractor.extract_features(DataLoader(dataset, batch_size=1)))

    assert len(results) == 1
    result = results[0]
    model_inputs = {
        "input_ids": encoded["input_ids"].to(model.device),
        "attention_mask": encoded["attention_mask"].to(model.device),
    }
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    embeddings = result.embeddings
    assert embeddings is not None
    assert torch.allclose(
        embeddings,
        outputs.hidden_states[0][0].detach().cpu(),
        atol=1e-4,
        rtol=1e-4,
    )

    attention_features = result.attention_features[0]
    query = attention_features.query
    key = attention_features.key
    value = attention_features.value
    attn_weights = attention_features.attn_weights
    assert query is not None and key is not None and value is not None
    assert attn_weights is not None

    architecture = get_model_architecture(model.__class__.__name__)
    if "position_ids" in model_inputs:
        position_ids = model_inputs["position_ids"][0].detach().cpu()
    else:
        sample_mask = model_inputs["attention_mask"][0].detach().cpu()
        position_ids = sample_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(sample_mask == 0, 0)
    layer_root = getattr(model, architecture.model_field, model)
    layer_module = getattr(getattr(layer_root, architecture.layer_field)[0], architecture.attn_field)
    qk_logits = reconstruct_attention_scores(
        architecture=architecture,
        query=query,
        key=key,
        layer_module=layer_module,
        rotary_emb_module=_resolve_rotary_module(model, architecture),
        position_ids=position_ids,
    )
    seq_len = qk_logits.shape[-1]
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=qk_logits.device, dtype=torch.bool),
        diagonal=1,
    )
    qk_logits = qk_logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
    if "attention_mask" in model_inputs:
        sample_mask = model_inputs["attention_mask"][0].detach().cpu().bool()
        qk_logits = qk_logits.masked_fill(~sample_mask.unsqueeze(0).unsqueeze(1), float("-inf"))
    expected_weights = torch.softmax(qk_logits, dim=-1)
    assert torch.allclose(expected_weights, attn_weights, atol=1e-4, rtol=1e-4)
    assert outputs.attentions is not None
    assert torch.allclose(
        attn_weights,
        outputs.attentions[0][0].detach().cpu(),
        atol=1e-4,
        rtol=1e-4,
    )

    merged_attn_output = attention_features.attn_weights @ value
    merged_attn_output = merged_attn_output.transpose(0, 1).reshape(value.shape[1], -1)
    layer_features = result.layer_features[0]
    assert layer_features.attn_output is not None
    assert torch.allclose(
        merged_attn_output,
        layer_features.attn_output,
        atol=1e-4,
        rtol=1e-4,
    )

    assert layer_features.output is not None
    final_hidden = outputs.hidden_states[-1][0].detach().cpu()
    assert not torch.allclose(
        layer_features.output,
        final_hidden,
        atol=1e-4,
        rtol=1e-4,
    )
    final_norm = _final_norm_module_from_model(model_name, model)
    final_norm_output = final_norm(layer_features.output.unsqueeze(0).to(model.device))[0]
    final_norm_output = final_norm_output.detach().cpu()
    assert torch.allclose(final_norm_output, final_hidden, atol=1e-4, rtol=1e-4)
