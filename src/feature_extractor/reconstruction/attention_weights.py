import math
from typing import Literal

import torch

from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HEAD, HEAD_DIM, SEQUENCE, Tensor


def create_causal_mask(
    sequence_length: int,
    mode: Literal["boolean", "additive"] = "boolean",
    dtype: torch.dtype | None = None,
) -> Tensor[SEQUENCE, SEQUENCE]:

    if mode == "boolean":
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length), diagonal=1
        ).bool()
    elif mode == "additive":
        assert dtype is not None, "dtype must be specified for additive mask"
        mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length), torch.finfo(dtype).min, dtype=dtype
            ),
            diagonal=1,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return mask


def apply_mask(
    attn_weights: Tensor[SEQUENCE, SEQUENCE],
    mask: Tensor[SEQUENCE, SEQUENCE],
    mask_value: float = float("nan"),
):
    attn_weights.shape[-1]
    mask = mask.to(attn_weights.device)
    attn_weights = attn_weights.masked_fill(mask, mask_value)
    return attn_weights


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    first = values[..., :half]
    second = values[..., half:]
    return torch.cat((-second, first), dim=-1)


def _reshape_rope_embeddings(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_dim = cos.dim()
    if cos_dim == 2:
        cos = cos[None, None, ...]
        sin = sin[None, None, ...]
    elif cos_dim == 3:
        cos = cos[:, None, ...]
        sin = sin[:, None, ...]
    elif cos_dim != 4:
        raise ValueError(
            "RoPE embeddings must have 2, 3, or 4 dimensions, got "
            f"{cos_dim}. Expected shapes like [seq_len, head_dim], "
            "[batch, seq_len, head_dim], or [batch, heads, seq_len, head_dim]."
        )

    head_dim = query.shape[-1]
    seq_len = query.shape[-2]
    cos = cos[..., :seq_len, :head_dim].to(dtype=query.dtype)
    sin = sin[..., :seq_len, :head_dim].to(dtype=query.dtype)
    return cos, sin


def _apply_rope(
    query: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    key: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    cos, sin = _reshape_rope_embeddings(cos, sin, query)
    query = query.to(cos.dtype).to(cos.device)
    key = key.to(cos.dtype).to(cos.device)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query, key


@torch.inference_mode()
def reconstruct_attention_weights(
    query: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    key: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    architecture: BaseModelArchitecture,
    before_softmax=False,
) -> Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]:
    """Reconstruct attention weights, matching GPT2 (SDPA) and Llama (RoPE) behavior.

    RoPE-based models upcast softmax to float32 in the transformers implementation
    for numerical stability. We mirror that behavior and cast back to the query
    dtype to match the model output.
    """
    use_rope = architecture.attn_use_rope

    if use_rope:
        if position_embeddings is None:
            raise ValueError("RoPE-based architectures require position embeddings.")
        query, key = _apply_rope(query, key, position_embeddings)

    attn_scores = torch.matmul(query, key.transpose(-1, -2))
    attn_scores = attn_scores / math.sqrt(query.shape[-1])

    if attention_mask is not None:
        attention_mask = attention_mask.to(attn_scores.dtype).to(attn_scores.device)
        attn_scores = attn_scores + attention_mask

    if before_softmax:
        return attn_scores.to(query.dtype)

    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights.to(query.dtype)
