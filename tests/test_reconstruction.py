from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from feature_extractor.models.architecture import (  # noqa: E402
    GPT2LMHeadModelArchitecture,
    LlamaForCausalLMArchitecture,
)
from feature_extractor.reconstruction import reconstruct_attention_scores  # noqa: E402


class _FakeRotary(nn.Module):
    def forward(self, x: torch.Tensor, *, position_ids: torch.Tensor):
        seq_len = x.shape[-2]
        head_dim = x.shape[-1]
        angles = position_ids.to(dtype=x.dtype).unsqueeze(-1).repeat(1, 1, head_dim)
        cos = torch.cos(angles)[:, :seq_len, :]
        sin = torch.sin(angles)[:, :seq_len, :]
        return cos, sin


class _FakeLlamaAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rotary_emb = _FakeRotary()


def test_reconstruct_attention_scores_gpt2_matches_scaled_dot_product() -> None:
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 8)

    logits = reconstruct_attention_scores(
        architecture=GPT2LMHeadModelArchitecture(),
        query=query,
        key=key,
    )

    expected = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(8)
    assert torch.allclose(logits, expected, atol=1e-6, rtol=1e-6)


def test_reconstruct_attention_scores_llama_applies_rotary_embedding() -> None:
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 3, 8)
    attn_module = _FakeLlamaAttention()
    position_ids = torch.tensor([0, 1, 2])

    logits = reconstruct_attention_scores(
        architecture=LlamaForCausalLMArchitecture(),
        query=query,
        key=key,
        layer_module=attn_module,
        position_ids=position_ids,
    )
    plain_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(8)

    assert logits.shape == plain_logits.shape
    assert not torch.allclose(logits, plain_logits)
