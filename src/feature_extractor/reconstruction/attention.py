import math

import torch

from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HEAD, HEAD_DIM, SEQUENCE, Tensor


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    first = values[..., :half]
    second = values[..., half:]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    head_dim = query.shape[-1]
    seq_len = query.shape[-2]

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.dim() != 4:
        raise ValueError(
            f"Unexpected RoPE embedding rank {cos.dim()}, expected 2-4."
        )

    cos = cos[..., :seq_len, :head_dim].to(dtype=query.dtype)
    sin = sin[..., :seq_len, :head_dim].to(dtype=query.dtype)

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
    if query is None or key is None:
        raise ValueError("query and key must be provided to reconstruct attention.")

    if architecture.attn_position_embeddings_arg_name is not None:
        if position_embeddings is None:
            raise ValueError("RoPE-based architectures require position embeddings.")
        query, key = _apply_rope(query, key, position_embeddings)

    attn_scores = torch.matmul(query, key.transpose(-1, -2))
    attn_scores = attn_scores / math.sqrt(query.shape[-1])

    mask_value = torch.finfo(attn_scores.dtype).min
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask
        mask_value = float(attention_mask.min().item())

    attn_scores = _apply_causal_mask(attn_scores, mask_value)

    if architecture.attn_position_embeddings_arg_name is not None:
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
    else:
        attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights.to(query.dtype)
