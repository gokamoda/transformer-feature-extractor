import math
from typing import Literal

import torch
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from feature_extractor.models import BaseModelArchitecture, load_causal_model
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


def get_o_proj_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model_name_or_path: str | None = None,
    model: PreTrainedModel | None = None,
):
    if model is not None:
        model_module = getattr(model, architecture.model_field)
    else:
        assert model_name_or_path is not None, (
            "Must provide either model or model_name_or_path"
        )
        model_module = getattr(
            load_causal_model(model_name_or_path), architecture.model_field
        )
    layer_module = getattr(model_module, architecture.layers_field)[layer_index]
    attn_module = getattr(layer_module, architecture.attn_field)
    o_proj_module = getattr(attn_module, architecture.attn_o_proj_field)
    return o_proj_module


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
    unfurl: Literal["none", "head_wise", "token_wise"] = "none",
    warnings_enabled: bool = True,
) -> (
    Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    | Tensor[BATCH, SEQUENCE, HEAD, HEAD_DIM]
    | Tensor[BATCH, SEQUENCE, HEAD, SEQUENCE, HIDDEN_DIM]
):
    assert unfurl in ["none", "head_wise", "token_wise"], (
        "unfurl must be one of 'none', 'head_wise', or 'token_wise'"
    )

    batch_size, num_heads, sequence_length, head_dim = value.shape

    # prepare weighted value for reconstruction
    if unfurl in ["none", "head_wise"]:
        weighted_value: Tensor[BATCH, SEQUENCE, HEAD, HEAD_DIM] = (
            torch.matmul(attn_weights, value).transpose(-2, -3).contiguous()
        )
    elif unfurl == "token_wise":
        weighted_value: Tensor[BATCH, SEQUENCE, HEAD, SEQUENCE, HEAD_DIM] = (
            torch.einsum(
                "bhij,bhjk->bhijk",
                attn_weights,  # [BATCH, HEAD, SEQUENCE, SEQUENCE]
                value,  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]
            )
            .transpose(-3, -4)
            .contiguous()
        )  # [BATCH, SEQUENCE, HEAD, SEQUENCE, HEAD_DIM]

    # prepare o_proj weight for reconstruction
    if unfurl in ["head_wise", "token_wise"]:
        if isinstance(o_proj_module, torch.nn.Linear):
            o_proj_weight_by_head = o_proj_module.weight.T.view(
                num_heads,
                head_dim,
                -1,  # output_dim // num_heads
            )
        elif isinstance(o_proj_module, Conv1D):
            o_proj_weight_by_head = o_proj_module.weight.view(
                num_heads,
                head_dim,
                -1,  # output_dim // num_heads
            )

    if unfurl == "none":
        concatenated_weighted_value_shape = (
            batch_size,
            sequence_length,
            -1,  # heads * head_dim
        )
        concatenated_weighted_value = weighted_value.reshape(
            concatenated_weighted_value_shape
        ).contiguous()
        attn_out_reconstructed = o_proj_module(concatenated_weighted_value)
    elif unfurl == "head_wise":
        attn_out_reconstructed: Tensor[BATCH, SEQUENCE, HEAD, HIDDEN_DIM] = (
            torch.einsum(
                "bshd,hdo->bsho",
                weighted_value,
                o_proj_weight_by_head,
            ).contiguous()
        )
        if warnings_enabled:
            print(
                "Take sum over dim -2 (head) and add bias of o_proj to reconstruct full attn output."
            )
    elif unfurl == "token_wise":
        attn_out_reconstructed: Tensor[BATCH, SEQUENCE, HEAD, SEQUENCE, HIDDEN_DIM] = (
            torch.einsum(
                "bihjd,hdo->bihjo",
                weighted_value,  # [BATCH, SEQUENCE, HEAD, SEQUENCE, HEAD_DIM]
                o_proj_weight_by_head,  # [HEAD, HEAD_DIM, OUTPUT_DIM // HEAD]
            ).contiguous()
        )
        if warnings_enabled:
            print(
                "Take sum over dim -2 (key), then over dim -2 (head) and add bias of o_proj to reconstruct full attn output."
            )

    return attn_out_reconstructed
