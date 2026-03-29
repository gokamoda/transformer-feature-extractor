import math

import torch
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from feature_extractor.models.architecture import BaseModelArchitecture

NEG_INF = -1e9


def _resolve_attn_module(
    model: PreTrainedModel, architecture: BaseModelArchitecture, layer_index: int
):
    layers_module = getattr(
        getattr(model, architecture.model_field),
        architecture.layers_field,
    )
    return getattr(layers_module[layer_index], architecture.attn_field)


def _build_attention_bias(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    batch_size = 1
    if attention_mask is not None:
        batch_size = attention_mask.shape[0]

    bias = torch.zeros((batch_size, 1, query_length, key_length), dtype=dtype, device=device)
    causal_mask = torch.triu(
        torch.full((query_length, key_length), NEG_INF, dtype=dtype, device=device),
        diagonal=1,
    )
    bias = bias + causal_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is None:
        return bias

    if attention_mask.dim() == 2:
        padding_mask = (1.0 - attention_mask.to(dtype=dtype)) * NEG_INF
        return bias + padding_mask[:, None, None, :]

    if attention_mask.dim() == 4:
        return bias + attention_mask.to(dtype=dtype)

    raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")


def _compute_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale: float,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    bias = _build_attention_bias(
        attention_mask,
        query_length=query.shape[-2],
        key_length=key.shape[-2],
        dtype=scores.dtype,
        device=scores.device,
    )
    scores = scores + bias
    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)


def reconstruct_gpt2_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    model: PreTrainedModel,
    architecture: BaseModelArchitecture,
    layer_index: int,
) -> torch.Tensor:
    attn_module = _resolve_attn_module(model, architecture, layer_index)
    head_dim = query.shape[-1]
    scale = 1.0

    if getattr(attn_module, "scale_attn_weights", True):
        scale /= math.sqrt(head_dim)

    if getattr(attn_module, "scale_attn_by_inverse_layer_idx", False):
        scale /= layer_index + 1

    return _compute_attention_weights(
        query,
        key,
        scale=scale,
        attention_mask=attention_mask,
    )


def reconstruct_llama_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    model: PreTrainedModel,
    architecture: BaseModelArchitecture,
    layer_index: int,
) -> torch.Tensor:
    attn_module = _resolve_attn_module(model, architecture, layer_index)

    cos, sin = attn_module.rotary_emb(key, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    return _compute_attention_weights(
        query,
        key,
        scale=1.0 / math.sqrt(query.shape[-1]),
        attention_mask=attention_mask,
    )


def reconstruct_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    model: PreTrainedModel,
    architecture: BaseModelArchitecture,
    layer_index: int,
) -> torch.Tensor:
    if architecture.absolute_pos_embedding_field is None:
        if position_ids is None:
            raise ValueError("position_ids are required for RoPE-based attention")
        return reconstruct_llama_attention_weights(
            query,
            key,
            attention_mask=attention_mask,
            position_ids=position_ids,
            model=model,
            architecture=architecture,
            layer_index=layer_index,
        )

    return reconstruct_gpt2_attention_weights(
        query,
        key,
        attention_mask=attention_mask,
        model=model,
        architecture=architecture,
        layer_index=layer_index,
    )
