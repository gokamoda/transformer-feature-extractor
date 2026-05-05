import math

import torch
from transformers.pytorch_utils import Conv1D

from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HEAD, HEAD_DIM, HIDDEN_DIM, SEQUENCE, Tensor

NON_ROPE_MASKED_BIAS = (
    -10000.0
)  # Matches GPT-2 masked_bias in transformers.modeling_gpt2.


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
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    cos, sin = _reshape_rope_embeddings(cos, sin, query)
    query = query.to(cos.dtype).to(cos.device)
    key = key.to(cos.dtype).to(cos.device)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query, key


def _apply_causal_mask(attn_scores: torch.Tensor, mask_value: float) -> torch.Tensor:
    seq_len = attn_scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=attn_scores.device, dtype=torch.bool),
        diagonal=1,
    )
    return attn_scores.masked_fill(causal_mask, mask_value)


def _get_mask_value(use_rope: bool, dtype: torch.dtype) -> float:
    if use_rope:
        return float(torch.finfo(dtype).min)
    return NON_ROPE_MASKED_BIAS


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


def create_causal_mask_single(
    sequence_length: int,
    dtype: torch.dtype,
):
    h = torch.full((sequence_length, sequence_length), torch.finfo(dtype).min)
    mask = torch.triu(h, diagonal=1)
    return mask


def reconstruct_attn_output(
    attn_weights: Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE],
    value: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
    o_proj_module: torch.nn.Module,
    head_wise: bool = False,
    warnings_enabled: bool = True,
) -> Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | Tensor[BATCH, SEQUENCE, HEAD, HEAD_DIM]:
    weighted_value: Tensor[BATCH, SEQUENCE, HEAD, HEAD_DIM] = (
        torch.matmul(attn_weights, value).transpose(-2, -3).contiguous()
    )

    if head_wise:
        if isinstance(o_proj_module, torch.nn.Linear):
            o_proj_weight_by_head = o_proj_module.weight.T.view(
                weighted_value.shape[2],  # head
                weighted_value.shape[3],  # head_dim
                -1,  # output_dim // num_heads
            )
        elif isinstance(o_proj_module, Conv1D):
            o_proj_weight_by_head = o_proj_module.weight.view(
                weighted_value.shape[2],  # head
                weighted_value.shape[3],  # head_dim
                -1,  # output_dim // num_heads
            )
        attn_out_reconstructed = torch.einsum(
            "bshd,hdo->bsho",
            weighted_value,
            o_proj_weight_by_head,
        )
        if warnings_enabled:
            print(
                "Take sum over dim -2 and add bias of o_proj to reconstruct full attn output."
            )
    else:
        concatenated_weighted_value_shape = (
            weighted_value.shape[0],  # batch
            weighted_value.shape[1],  # sequence
            -1,  # heads * head_dim
        )
        concatenated_weighted_value = weighted_value.reshape(
            concatenated_weighted_value_shape
        ).contiguous()
        attn_out_reconstructed = o_proj_module(concatenated_weighted_value)
    return attn_out_reconstructed
