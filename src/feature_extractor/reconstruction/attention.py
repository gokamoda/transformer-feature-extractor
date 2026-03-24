from __future__ import annotations

import math

import torch
from torch import nn

from feature_extractor.models.architecture import BaseModelArchitecture


def reconstruct_attention_scores(
    *,
    architecture: BaseModelArchitecture,
    query: torch.Tensor,
    key: torch.Tensor,
    layer_module: nn.Module | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct pre-softmax attention scores from captured q/k projections."""

    if query.dim() != 3 or key.dim() != 3:
        msg = (
            "Query and key tensors must be 3D with shape "
            "(heads, sequence, head_dim)."
        )
        raise ValueError(msg)
    if query.shape[-1] != key.shape[-1]:
        msg = (
            "Query/key head dimensions must match for score reconstruction "
            f"(got {query.shape[-1]} and {key.shape[-1]})."
        )
        raise ValueError(msg)

    query_proj = query
    key_proj = key
    if architecture.attn_field == "self_attn":
        query_proj, key_proj = _apply_llama_rotary_embeddings(
            query,
            key,
            layer_module=layer_module,
            position_ids=position_ids,
        )

    scale = 1.0 / math.sqrt(query_proj.shape[-1])
    return torch.matmul(query_proj, key_proj.transpose(-2, -1)) * scale


def _apply_llama_rotary_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    layer_module: nn.Module | None,
    position_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_module is None or not hasattr(layer_module, "rotary_emb"):
        msg = "Llama-style attention score reconstruction requires rotary_emb module."
        raise ValueError(msg)
    if position_ids is None:
        msg = "Llama-style attention score reconstruction requires position_ids."
        raise ValueError(msg)

    batch_query = query.unsqueeze(0)
    batch_key = key.unsqueeze(0)
    batch_positions = position_ids.unsqueeze(0)

    rotary_emb = layer_module.rotary_emb
    rotary_output = rotary_emb(batch_query, position_ids=batch_positions)
    if isinstance(rotary_output, tuple):
        cos, sin = rotary_output
    else:
        cos = getattr(rotary_output, "cos", None)
        sin = getattr(rotary_output, "sin", None)
    if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
        msg = "Unable to resolve cosine/sine tensors from rotary embedding module."
        raise ValueError(msg)

    query_rot = _apply_rotary_pos_emb(batch_query, cos, sin)
    key_rot = _apply_rotary_pos_emb(batch_key, cos, sin)
    return query_rot[0], key_rot[0]


def _apply_rotary_pos_emb(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    while cos.dim() < tensor.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (tensor * cos) + (_rotate_half(tensor) * sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)
