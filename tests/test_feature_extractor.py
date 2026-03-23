from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.extractor.base import BaseFeatureExtractor
from feature_extractor.models.architecture import (
    BaseModelArchitecture,
    QKV_IMPLEMENTATION_CONV1D,
)


def _patch_model_and_tokenizer(monkeypatch, model, tokenizer) -> None:
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )


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


class DummyTensorAttentionModel(nn.Module):
    def __init__(
        self, hidden_size: int = 4, num_layers: int = 2, num_heads: int = 1
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.num_heads = num_heads
        self.last_output_attentions: bool | None = None

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
        self.last_output_attentions = output_attentions
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
            attentions = torch.stack(attention_layers, dim=0)
        return SimpleNamespace(hidden_states=tuple(hidden_states), attentions=attentions)


class DummySDPAModel(DummyModel):
    def __init__(
        self, hidden_size: int = 4, num_layers: int = 2, num_heads: int = 1
    ) -> None:
        super().__init__(
            hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads
        )
        self.config = SimpleNamespace(attn_implementation="sdpa")

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
        # Allow eager overrides to bypass the SDPA error.
        if self.config.attn_implementation == "sdpa" and output_attentions:
            raise ValueError(
                "SDPA attention does not support output_attentions=True"
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class StaticSDPAConfig:
    def __init__(self) -> None:
        self._attn_implementation = "sdpa"

    @property
    def attn_implementation(self) -> str:
        return self._attn_implementation

    @attn_implementation.setter
    def attn_implementation(self, _value: str) -> None:
        return None


class DummySDPAFallbackModel(DummyModel):
    def __init__(
        self, hidden_size: int = 4, num_layers: int = 2, num_heads: int = 1
    ) -> None:
        super().__init__(
            hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads
        )
        self.config = StaticSDPAConfig()
        self.sdpa_error_count = 0

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
        if self.config.attn_implementation == "sdpa" and output_attentions:
            self.sdpa_error_count += 1
            raise ValueError(
                "SDPA attention does not support output_attentions=True"
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class DummyMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = torch.relu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(hidden_states)))


class DummyMLPLayer(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int) -> None:
        super().__init__()
        self.mlp = DummyMLP(hidden_size, mlp_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class DummyMLPModel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, hidden_size)
        self.layers = nn.ModuleList(
            [DummyMLPLayer(hidden_size, mlp_dim) for _ in range(num_layers)]
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
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(hidden_states), attentions=None)


def _compute_expected_mlp_activation(
    model: DummyMLPModel,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
) -> torch.Tensor:
    hidden = model.embedding(input_ids)
    return model.layers[layer_idx].mlp.act_fn(model.layers[layer_idx].mlp.fc1(hidden))


class DummyLlamaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_qk_logits: torch.Tensor | None = None
        self.attn_weights: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_proj = self.q_proj(hidden_states)
        key_proj = self.k_proj(hidden_states)
        value_proj = self.v_proj(hidden_states)
        batch_size, seq_len, _ = query_proj.shape
        query = query_proj.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key_proj.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if self.num_heads != self.num_key_value_heads:
            key = key.repeat_interleave(
                self.num_heads // self.num_key_value_heads, dim=1
            )
        self.last_qk_logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        self.attn_weights = torch.softmax(self.last_qk_logits, dim=-1)
        combined = query_proj + key_proj.mean(dim=-1, keepdim=True) + value_proj.mean(
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


class DummyGPT2Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.c_attn = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        return self.c_proj(query + key + value)


class DummyGPT2Layer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = DummyGPT2Attention(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden_states)


class DummyGPT2Inner(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.h = nn.ModuleList([DummyGPT2Layer(hidden_size)])


class DummyGPT2Model(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, hidden_size)
        self.transformer = DummyGPT2Inner(hidden_size)
        self.config = SimpleNamespace(n_head=num_heads, n_embd=hidden_size)

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
        for layer in self.transformer.h:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(hidden_states), attentions=None)

def test_extract_features_embeddings_and_residual(monkeypatch):
    model = DummyModel(hidden_size=4, num_layers=2)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

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
    assert results[0].layer_features[0].layer_index == 0
    assert results[0].layer_features[1].layer_index == 1
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

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

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

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

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
    assert results[0].layer_features[0].layer_index == 0
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

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

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
    assert results[0].attention_features[0].layer_index == 0
    assert results[0].attention_features[0].attn_weights.shape == (1, 3, 3)


def test_extract_features_with_tensor_attentions(monkeypatch):
    model = DummyTensorAttentionModel(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(feature_names=["attn.layer_00.weights"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert model.last_output_attentions is True
    weights = results[0].attention_features[0].attn_weights
    assert weights is not None
    assert weights.shape == (1, 3, 3)


def test_extract_features_with_sdpa_attention_override(monkeypatch):
    model = DummySDPAModel(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)
    seen_attn_implementations: list[str] = []
    original_forward = model.forward

    def _wrapped_forward(*args, **kwargs):
        seen_attn_implementations.append(model.config.attn_implementation)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", _wrapped_forward)

    feature_cfg = FeatureConfig(feature_names=["attn.layer_00.weights"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert model.last_output_attentions is True
    assert seen_attn_implementations
    assert all(
        implementation == "eager" for implementation in seen_attn_implementations
    )
    assert model.config.attn_implementation == "sdpa"
    weights = results[0].attention_features[0].attn_weights
    assert weights is not None
    assert weights.shape == (1, 3, 3)


def test_extract_features_with_sdpa_attention_fallback(monkeypatch):
    model = DummySDPAFallbackModel(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(feature_names=["attn.layer_00.weights"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert model.sdpa_error_count == 1
    assert model.last_output_attentions is False
    assert results[0].attention_features[0].attn_weights is None


def test_extract_features_with_attention_weights_fallback(monkeypatch):
    model = DummyLlamaModel(hidden_size=4, num_heads=2, num_key_value_heads=1)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(feature_names=["attn.layer_00.weights"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    weights = results[0].attention_features[0].attn_weights
    assert weights is not None
    assert weights.shape == (2, 3, 3)
    assert model.model.layers[0].self_attn.attn_weights is not None
    expected_weights = model.model.layers[0].self_attn.attn_weights[0].detach().cpu()
    assert torch.allclose(weights, expected_weights)


def test_extract_features_with_layer_attn_output(monkeypatch):
    model = DummyLlamaModel(hidden_size=4, num_heads=2, num_key_value_heads=1)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(feature_names=["layer.layer_00.attn_output"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert results[0].layer_features[0].attn_output is not None
    assert results[0].layer_features[0].attn_output.shape == (3, 4)

    with torch.no_grad():
        model_output = model(
            input_ids=torch.stack([item["input_ids"] for item in dataset])
        )
    assert torch.allclose(
        results[0].layer_features[0].attn_output, model_output.hidden_states[1][0]
    )


def test_extract_features_with_qkv_gqa(monkeypatch):
    model = DummyLlamaModel(hidden_size=4, num_heads=2, num_key_value_heads=1)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

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
    assert features.layer_index == 0
    assert features.query.shape == (2, 3, 2)
    assert features.key.shape == (2, 3, 2)
    assert features.value.shape == (2, 3, 2)
    assert features.qk_logits.shape == (2, 3, 3)


def test_extract_features_with_conv1d_qkv(monkeypatch):
    model = DummyGPT2Model(hidden_size=4, num_heads=2)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)
    monkeypatch.setattr(
        "feature_extractor.extractor.base.get_model_architecture",
        lambda _: BaseModelArchitecture(
            model_field="transformer",
            layer_field="h",
            attn_field="attn",
            qkv_implementation=QKV_IMPLEMENTATION_CONV1D,
        ),
    )

    feature_cfg = FeatureConfig(feature_names=["attn.layer_00.query"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    features = results[0].attention_features[0]
    assert features.query is not None
    assert features.query.shape == (2, 3, 2)


def test_extract_features_with_mlp_activation(monkeypatch):
    model = DummyMLPModel(hidden_size=4, mlp_dim=6, num_layers=1)
    tokenizer = DummyTokenizer()

    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(feature_names=["mlp.layer_00.activation"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    dataset = [
        {"idx": "a", "input_ids": torch.tensor([1, 2, 3], dtype=torch.long)}
    ]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert len(results[0].mlp_features) == 1
    activation = results[0].mlp_features[0].activation
    assert activation is not None
    assert activation.shape == (3, 6)

    with torch.no_grad():
        expected = _compute_expected_mlp_activation(
            model,
            dataset[0]["input_ids"].unsqueeze(0),
        )[0]
    assert torch.allclose(activation, expected)
