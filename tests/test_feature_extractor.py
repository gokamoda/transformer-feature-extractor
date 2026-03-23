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
    def __init__(
        self, hidden_size: int = 4, num_layers: int = 2, num_heads: int = 1
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.num_heads = num_heads
        self.last_attention_mask: torch.Tensor | None = None
        self.last_output_attentions: bool | None = None
        self.extra_keys: tuple[str, ...] = ()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        self.last_attention_mask = attention_mask
        self.last_output_attentions = output_attentions
        self.extra_keys = tuple(kwargs.keys())
        hidden_states = []
        hidden = self.embedding(input_ids)
        hidden_states.append(hidden)
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        attentions = None
        if output_attentions:
            batch_size, seq_len = input_ids.shape
            eye = torch.eye(seq_len, dtype=hidden.dtype, device=hidden.device)
            attn = eye.unsqueeze(0).unsqueeze(0).repeat(
                batch_size, self.num_heads, 1, 1
            )
            attention_layers = [attn.clone() for _ in self.layers]
            attentions = tuple(attention_layers)
        return SimpleNamespace(hidden_states=tuple(hidden_states), attentions=attentions)


class DummyLlamaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        combined = query + key.mean(dim=-1, keepdim=True) + value.mean(
            dim=-1, keepdim=True
        )
        return self.o_proj(combined)


class DummyLlamaLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        self.self_attn = DummyLlamaAttention(hidden_size, num_heads, num_key_value_heads)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.self_attn(hidden_states)


class DummyLlamaInner(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DummyLlamaLayer(hidden_size, num_heads, num_key_value_heads)]
        )


class DummyLlamaModel(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, hidden_size)
        self.model = DummyLlamaInner(hidden_size, num_heads, num_key_value_heads)
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        hidden_states = []
        hidden = self.embedding(input_ids)
        hidden_states.append(hidden)
        for layer in self.model.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(hidden_states), attentions=None)


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
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)},
        {"idx": "b", "input_ids": torch.tensor([4, 5, 6], dtype=torch.long)},
    ]
    data_loader = DataLoader(dataset, batch_size=2)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 2
    assert model.extra_keys == ()
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

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 2
    assert model.last_attention_mask is not None
    assert torch.equal(
        model.last_attention_mask,
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    )


def test_extract_features_layer_outputs(monkeypatch):
    model = DummyModel(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )

    feature_cfg = FeatureConfig(
        feature_names=["layer.layer_00.ffn_output", "layer.layer_00.output"]
    )
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert len(results[0].layer_features) == 1
    assert results[0].layer_features[0].mlp_output is not None
    assert results[0].layer_features[0].output is not None

    with torch.no_grad():
        model_output = model(
            input_ids=torch.stack([item["input_ids"] for item in dataset])
        )
    assert torch.allclose(
        results[0].layer_features[0].mlp_output, model_output.hidden_states[1][0]
    )
    assert torch.allclose(
        results[0].layer_features[0].output, model_output.hidden_states[1][0]
    )


def test_extract_features_with_attention_weights(monkeypatch):
    model = DummyModel(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )

    feature_cfg = FeatureConfig(feature_names=["attn.layer_00.weights"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert model.last_output_attentions is True
    assert len(results[0].attention_features) == 1
    assert results[0].attention_features[0].attn_weights.shape == (1, 3, 3)


def test_extract_features_with_qkv_gqa(monkeypatch):
    model = DummyLlamaModel(hidden_size=4, num_heads=2, num_key_value_heads=1)
    tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )

    feature_cfg = FeatureConfig(
        feature_names=[
            "attn.layer_00.query",
            "attn.layer_00.key",
            "attn.layer_00.value",
            "attn.layer_00.qk_logits",
        ]
    )
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert len(results[0].attention_features) == 1
    features = results[0].attention_features[0]
    assert features.query.shape == (2, 3, 2)
    assert features.key.shape == (2, 3, 2)
    assert features.value.shape == (2, 3, 2)
    assert features.qk_logits.shape == (2, 3, 3)
