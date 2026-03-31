import math

import torch

from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HEAD, HEAD_DIM, SEQUENCE, Tensor

GPT2_MASKED_BIAS = -10000.0


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
    unsqueeze_dims_by_rank = {2: (0, 0), 3: (1,)}
    if cos_dim in unsqueeze_dims_by_rank:
        for dim in unsqueeze_dims_by_rank[cos_dim]:
            cos = cos.unsqueeze(dim)
            sin = sin.unsqueeze(dim)
    elif cos_dim != 4:
        raise ValueError(
            "RoPE embeddings must have 2-4 dimensions, got "
            f"{cos_dim}. Ensure position_embeddings are properly formatted."
        )

    head_dim = query.shape[-1]
    seq_len = query.shape[-2]
    cos = cos[..., :seq_len, :head_dim].to(dtype=query.dtype)
    sin = sin[..., :seq_len, :head_dim].to(dtype=query.dtype)
    return cos, sin


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    cos, sin = _reshape_rope_embeddings(cos, sin, query)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query, key


def _apply_causal_mask(
    attn_scores: torch.Tensor, mask_value: float
) -> torch.Tensor:
    seq_len = attn_scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=attn_scores.device, dtype=torch.bool),
        diagonal=1,
    )
    return attn_scores.masked_fill(causal_mask, mask_value)


def reconstruct_attention_weights(
    query: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    key: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    architecture: BaseModelArchitecture,
) -> Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]:
    """Reconstruct attention weights, matching GPT2 (SDPA) and Llama (RoPE) behavior.

    RoPE-based models upcast softmax to float32 in the transformers implementation,
    so we mirror that behavior and cast back to the query dtype.
    """
    if architecture.attn_position_embeddings_arg_name is not None:
        if position_embeddings is None:
            raise ValueError("RoPE-based architectures require position embeddings.")
        query, key = _apply_rope(query, key, position_embeddings)

    attn_scores = torch.matmul(query, key.transpose(-1, -2))
    attn_scores = attn_scores / math.sqrt(query.shape[-1])

    if architecture.attn_position_embeddings_arg_name is not None:
        mask_value = float(torch.finfo(attn_scores.dtype).min)
    else:
        mask_value = GPT2_MASKED_BIAS

    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    attn_scores = _apply_causal_mask(attn_scores, mask_value)

    if architecture.attn_position_embeddings_arg_name is not None:
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
    else:
        attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights.to(query.dtype)
