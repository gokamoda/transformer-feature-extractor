from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.extractor.base import BaseFeatureExtractor
from feature_extractor.models.architecture import BaseModelArchitecture

DUMMY_VOCAB_SIZE = 20  # Shared dummy vocab size for tokenizer + embedding layers.


def _patch_model_and_tokenizer(monkeypatch, model, tokenizer) -> None:
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_causal_model", lambda _: model
    )
    monkeypatch.setattr(
        "feature_extractor.extractor.base.load_tokenizer", lambda _: tokenizer
    )


class DummyTokenizer:
    """Minimal tokenizer for tests that maps tokens to stable hashed IDs."""

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        max_len = max(len(text.split()) for text in texts)
        input_ids = []
        for text in texts:
            token_ids = [_stable_token_id(token) for token in text.split()]
            token_ids += [0] * (max_len - len(token_ids))
            input_ids.append(token_ids)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}


def _stable_token_id(token: str) -> int:
    """Return a stable token ID using SHA-256 and mod DUMMY_VOCAB_SIZE."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % DUMMY_VOCAB_SIZE


class DummyModel(nn.Module):
    def __init__(
        self, hidden_size: int = 4, num_layers: int = 1, num_heads: int = 1
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(DUMMY_VOCAB_SIZE, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.num_heads = num_heads

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


class DummyProperAttnModule(nn.Module):
    """Multi-head attention (no output projection) for feature extraction tests.

    Computes standard scaled-dot-product attention: output = softmax(QK^T/sqrt(d)) @ V.
    Stores attention weights for the most recent forward pass on
    ``self.attn_weights`` so hooks can retrieve them via the attention hook
    manager.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(b, t, h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, d).transpose(1, 2)
        self.attn_weights = torch.softmax(
            q @ k.transpose(-2, -1) * self.scale, dim=-1
        )  # (B, H, T, T)
        context = self.attn_weights @ v  # (B, H, T, D)
        # Return weighted-value sum (no output projection) reshaped to (B, T, C)
        return context.transpose(1, 2).contiguous().view(b, t, h * d)


class DummyProperAttnLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = DummyProperAttnModule(hidden_size, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


class DummyProperAttnInner(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DummyProperAttnLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )


class DummyLlamaProperAttnModel(nn.Module):
    """Llama-shaped model with proper attention for feature-correctness tests.

    Uses the architecture fallback (``BaseModelArchitecture``) which resolves
    ``model.model.layers[*].self_attn`` with ``q_proj/k_proj/v_proj`` through
    the default layer/attention field fallback logic in
    ``feature_extractor.hooks.attention.AttentionHookManager``.
    """

    def __init__(
        self, hidden_size: int = 8, num_heads: int = 2, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(DUMMY_VOCAB_SIZE, hidden_size)
        self.model = DummyProperAttnInner(hidden_size, num_heads, num_layers)
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
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
        hidden = self.embedding(input_ids)
        all_hidden_states = [hidden]
        all_attentions = []
        for layer in self.model.layers:
            hidden = layer(hidden)
            all_hidden_states.append(hidden)
            if output_attentions:
                all_attentions.append(layer.self_attn.attn_weights)
        return SimpleNamespace(
            hidden_states=tuple(all_hidden_states),
            attentions=tuple(all_attentions) if output_attentions else None,
        )


class DummyModelWithFinalNorm(nn.Module):
    """Model with a final LayerNorm applied after all transformer layers.

    ``output_hidden_states`` contains raw (pre-norm) per-layer outputs.
    ``last_hidden_state`` holds the final normalized representation.
    """

    def __init__(self, hidden_size: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(DUMMY_VOCAB_SIZE, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(hidden_size)

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
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        last_hidden_state = self.ln_final(hidden)
        return SimpleNamespace(
            hidden_states=tuple(hidden_states),
            last_hidden_state=last_hidden_state,
            attentions=None,
        )


def test_embedding_is_close_to_output_hidden_states(monkeypatch):
    """Embeddings extracted by the feature extractor equal model hidden_states[0]."""
    model = DummyModel(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()
    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(feature_names=["embeddings"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    dataset = [{"input_ids": input_ids}]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    with torch.no_grad():
        model_output = model(input_ids=input_ids.unsqueeze(0))

    assert results[0].embeddings is not None
    assert torch.allclose(results[0].embeddings, model_output.hidden_states[0][0])


def test_qk_vectors_yield_attention_weights(monkeypatch):
    """Q,K captured by hooks reproduce the attention weights from output_attentions."""
    model = DummyLlamaProperAttnModel(hidden_size=8, num_heads=2, num_layers=1)
    tokenizer = DummyTokenizer()
    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(
        feature_names=["attn.layer_00.query", "attn.layer_00.key"]
    )
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    dataset = [{"input_ids": input_ids}]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    # Retrieve ground-truth attention weights via output_attentions
    with torch.no_grad():
        model_output = model(
            input_ids=input_ids.unsqueeze(0), output_attentions=True
        )

    query = results[0].attention_features[0].query  # (num_heads, seq, head_dim)
    key = results[0].attention_features[0].key  # (num_heads, seq, head_dim)
    head_dim = model.model.layers[0].self_attn.head_dim
    scale = head_dim**-0.5

    # Compute attention weights from extracted Q and K
    computed_weights = torch.softmax(
        query @ key.transpose(-2, -1) * scale, dim=-1
    )  # (num_heads, seq, seq)

    # output_attentions[layer=0][sample=0]: (num_heads, seq, seq)
    expected_weights = model_output.attentions[0][0].detach().cpu()

    assert computed_weights.shape == expected_weights.shape
    assert torch.allclose(computed_weights, expected_weights, atol=1e-6)


def test_value_vectors_yield_attention_output(monkeypatch):
    """Weighted sum of captured V vectors matches the attention output from hooks."""
    model = DummyLlamaProperAttnModel(hidden_size=8, num_heads=2, num_layers=1)
    tokenizer = DummyTokenizer()
    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)

    feature_cfg = FeatureConfig(
        feature_names=[
            "attn.layer_00.value",
            "attn.layer_00.weights",
            "layer.layer_00.attn_output",
        ]
    )
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    dataset = [{"input_ids": input_ids}]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    value = results[0].attention_features[0].value  # (num_heads, seq, head_dim)
    attn_weights = results[0].attention_features[0].attn_weights  # (num_heads, seq, seq)
    attn_output = results[0].layer_features[0].attn_output  # (seq, hidden_size)

    assert value is not None
    assert attn_weights is not None
    assert attn_output is not None

    # context = attn_weights @ V: (num_heads, seq, head_dim)
    # Reshape to (seq, hidden_size) to match attn_output
    seq_len = value.shape[1]
    hidden_size = value.shape[0] * value.shape[2]
    computed_output = (attn_weights @ value).transpose(0, 1).reshape(
        seq_len, hidden_size
    )

    assert torch.allclose(computed_output, attn_output, atol=1e-6)


def test_last_hidden_state_differs_from_ln_final_output(monkeypatch):
    """Last layer hidden state (pre-ln_final) is not equal to model's last_hidden_state."""
    model = DummyModelWithFinalNorm(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()
    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)
    monkeypatch.setattr(
        "feature_extractor.extractor.base.get_model_architecture",
        lambda _: BaseModelArchitecture(),
    )

    feature_cfg = FeatureConfig(feature_names=["layer.layer_00.output"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    dataset = [{"input_ids": input_ids}]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    with torch.no_grad():
        model_output = model(input_ids=input_ids.unsqueeze(0))

    last_layer_output = results[0].layer_features[0].output  # pre-ln_final
    last_hidden_state = model_output.last_hidden_state[0]  # post-ln_final

    # LayerNorm changes the values, so they should NOT be equal
    assert not torch.allclose(last_layer_output, last_hidden_state)


def test_last_hidden_state_plus_ln_final_matches_model_output(monkeypatch):
    """Applying ln_final to the captured last layer output reproduces last_hidden_state."""
    model = DummyModelWithFinalNorm(hidden_size=4, num_layers=1)
    tokenizer = DummyTokenizer()
    _patch_model_and_tokenizer(monkeypatch, model, tokenizer)
    monkeypatch.setattr(
        "feature_extractor.extractor.base.get_model_architecture",
        lambda _: BaseModelArchitecture(),
    )

    feature_cfg = FeatureConfig(feature_names=["layer.layer_00.output"])
    extractor = BaseFeatureExtractor("dummy", feature_cfg)
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    dataset = [{"input_ids": input_ids}]
    data_loader = DataLoader(dataset, batch_size=1)

    results = list(extractor.extract_features(data_loader))

    with torch.no_grad():
        model_output = model(input_ids=input_ids.unsqueeze(0))

    last_layer_output = results[0].layer_features[0].output  # pre-ln_final
    last_hidden_state = model_output.last_hidden_state[0]  # post-ln_final

    # Applying the model's final norm to the captured output must reproduce last_hidden_state
    with torch.no_grad():
        normalized = model.ln_final(last_layer_output)

    assert torch.allclose(normalized, last_hidden_state, atol=1e-6)
