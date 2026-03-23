from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.extractor.base import BaseFeatureExtractor


class DummyTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        max_len = max(len(text.split()) for text in texts)
        input_ids = []
        for text in texts:
            token_ids = [len(token) % 20 for token in text.split()]
            token_ids += [0] * (max_len - len(token_ids))
            input_ids.append(token_ids)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}


class DummyModel(nn.Module):
    def __init__(self, hidden_size: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.last_attention_mask: torch.Tensor | None = None

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        self.last_attention_mask = attention_mask
        hidden_states = []
        hidden = self.embedding(input_ids)
        hidden_states.append(hidden)
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


def test_extract_features_embeddings_and_residual(monkeypatch):
    model = DummyModel(hidden_size=4, num_layers=2)
    tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )

    feature_cfg = FeatureConfig(
        feature_names=[
            "embeddings",
            "residual.layer_00.pre_attn",
            "residual.layer_01.post_ffn",
        ]
    )
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"input_ids": torch.tensor([1, 2, 3], dtype=torch.long)},
        {"input_ids": torch.tensor([4, 5, 6], dtype=torch.long)},
    ]
    data_loader = DataLoader(dataset, batch_size=2)

    results = extractor.extract_features(data_loader)

    assert len(results) == 2
    assert results[0].embeddings.shape == (3, 4)
    assert len(results[0].layer_features) == 2
    assert results[0].layer_features[0].input is not None
    assert results[0].layer_features[0].output is None
    assert results[0].layer_features[1].input is None
    assert results[0].layer_features[1].output is not None

    with torch.no_grad():
        model_output = model(
            input_ids=torch.stack([item["input_ids"] for item in dataset])
        )

    assert torch.allclose(results[0].embeddings, model_output.hidden_states[0][0])
    assert torch.allclose(
        results[0].layer_features[0].input, model_output.hidden_states[0][0]
    )
    assert torch.allclose(
        results[0].layer_features[1].output, model_output.hidden_states[2][0]
    )


def test_extract_features_with_attention_mask(monkeypatch):
    model = DummyModel(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )

    feature_cfg = FeatureConfig(feature_names=["embeddings"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    # Use tuples to exercise tensor tuple handling in _prepare_batch.
    dataset = [
        (
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([1, 1, 1], dtype=torch.long),
        ),
        (
            torch.tensor([4, 5, 6], dtype=torch.long),
            torch.tensor([1, 1, 0], dtype=torch.long),
        ),
    ]
    data_loader = DataLoader(dataset, batch_size=2)

    results = extractor.extract_features(data_loader)

    assert len(results) == 2
    assert model.last_attention_mask is not None
    assert torch.equal(
        model.last_attention_mask,
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    )
