import math
from typing import Literal

import torch
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from feature_extractor.models import BaseModelArchitecture, load_causal_model
from feature_extractor.models.architecture import (
    QKV_IMPLEMENTATION_CONV1D,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
)
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


def get_v_proj_module(
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

    if architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
        assert architecture.attn_v_proj_field is not None, (
            "Must specify attn_v_proj_field in architecture to get v_proj module."
        )
        v_proj_module = getattr(attn_module, architecture.attn_v_proj_field)
    elif architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
        assert architecture.attn_qkv_proj_field is not None, (
            "Must specify attn_qkv_proj_field in architecture when using conv1d qkv implementation."
        )
        qkv_proj_module = getattr(attn_module, architecture.attn_qkv_proj_field)

        in_features = qkv_proj_module.nx
        out_features = (
            qkv_proj_module.nf // 3
        )  # Assuming q, k, v are concatenated along the output dimension
        if hasattr(qkv_proj_module, "bias") and qkv_proj_module.bias is not None:
            has_bias = True
        else:
            has_bias = False

        v_proj_module = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        )
        v_proj_module.weight = torch.nn.Parameter(
            qkv_proj_module.weight[
                :, 2 * out_features : 3 * out_features
            ].T.contiguous()
        )
        if has_bias:
            v_proj_module.bias = torch.nn.Parameter(
                qkv_proj_module.bias[2 * out_features : 3 * out_features]
            )

    return v_proj_module


def get_pre_attn_norm_module(
    architecture: BaseModelArchitecture,
    layer_index: int,
    model_name_or_path: str | None = None,
    model: PreTrainedModel | None = None,
):
    assert architecture.pre_attn_ln_field is not None, (
        "Architecture does not specify a pre-attention layer norm field."
    )
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
    return getattr(layer_module, architecture.pre_attn_ln_field)


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
    o_proj_module: torch.nn.Linear | Conv1D,
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
        o_proj_weight_by_head = _split_o_proj_by_head(
            o_proj_module=o_proj_module,
            head_dim=head_dim,
            num_attention_heads=num_heads,
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


def _split_v_proj_by_head(
    v_proj_module: torch.nn.Linear,
    head_dim: int,
    num_attention_heads: int,
    num_kv_heads: int,
) -> tuple[Tensor[HEAD, HIDDEN_DIM, HEAD_DIM], Tensor[HEAD, HEAD_DIM] | None]:
    n_repeat = num_attention_heads // num_kv_heads
    v_proj_weight_by_head: Tensor[HEAD, HIDDEN_DIM, HEAD_DIM] = (
        v_proj_module.weight.T.view(
            -1,  # output_dim // num_heads
            num_kv_heads,
            head_dim,
        )
        .transpose(0, 1)
        .contiguous()
    )  # [HEAD, HIDDEN_DIM, HEAD_DIM]
    v_proj_weight_by_head = v_proj_weight_by_head.repeat_interleave(n_repeat, dim=0)

    v_proj_bias_by_head: Tensor[HEAD, HEAD_DIM] | None = None
    if v_proj_module.bias is not None:
        v_proj_bias_by_head = v_proj_module.bias.view(
            num_kv_heads,
            head_dim,
        ).repeat_interleave(n_repeat, dim=0)

    return v_proj_weight_by_head, v_proj_bias_by_head


def _split_o_proj_by_head(
    o_proj_module: torch.nn.Linear | Conv1D,
    head_dim: int,
    num_attention_heads: int,
) -> Tensor[HEAD, HEAD_DIM, HIDDEN_DIM]:
    if isinstance(o_proj_module, torch.nn.Linear):
        o_proj_weight_by_head = o_proj_module.weight.T.view(
            num_attention_heads,
            head_dim,
            -1,  # output_dim // num_heads
        )
    elif isinstance(o_proj_module, Conv1D):
        o_proj_weight_by_head = o_proj_module.weight.view(
            num_attention_heads,
            head_dim,
            -1,  # output_dim // num_heads
        )
    return o_proj_weight_by_head


def reconstruct_value_vectors(
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    v_proj_module: torch.nn.Module,
    num_kv_heads: int,
    num_attention_heads: int,
) -> Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]:

    assert isinstance(v_proj_module, torch.nn.Linear), (
        "Currently only supports linear v_proj modules."
    )
    head_dim = v_proj_module.out_features // num_kv_heads
    v_proj_weight_by_head, v_proj_bias_by_head = _split_v_proj_by_head(
        v_proj_module=v_proj_module,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
    )

    value_vectors: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] = torch.einsum(
        "bid,hde->bihe",
        hidden_states,  # [BATCH, SEQUENCE, HIDDEN_DIM]
        v_proj_weight_by_head,  # [HEAD, HEAD_DIM, HIDDEN_DIM // HEAD]
    ).contiguous()  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]

    if v_proj_bias_by_head is not None:
        value_vectors = value_vectors + v_proj_bias_by_head

    return value_vectors.transpose(
        1, 2
    )  # [BATCH, SEQUENCE, HEAD, HEAD_DIM] -> [BATCH, HEAD, SEQUENCE, HEAD_DIM]


def _precompute_ov_weights(
    v_proj_module: torch.nn.Linear,
    o_proj_module: torch.nn.Linear | Conv1D,
    num_attention_heads: int,
    head_dim: int,
    num_kv_heads: int,
):
    v_proj_weight_by_head, v_proj_bias_by_head = _split_v_proj_by_head(
        v_proj_module=v_proj_module,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
    )
    # o_proj
    o_proj_weight_by_head = _split_o_proj_by_head(
        o_proj_module=o_proj_module,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
    )

    ov_combined_weight_by_head = torch.einsum(
        "hde,heo->hdo",
        v_proj_weight_by_head,  # [HEAD, HEAD_DIM, HIDDEN_DIM // HEAD]
        o_proj_weight_by_head,  # [HEAD, HEAD_DIM, HIDDEN_DIM // HEAD]
    ).contiguous()  # [HEAD, HEAD, HIDDEN_DIM // HEAD]

    bias = None
    if v_proj_bias_by_head is not None:
        bias = torch.einsum(
            "he,heo->ho",
            v_proj_bias_by_head,  # [HEAD, HEAD_DIM]
            o_proj_weight_by_head,  # [HEAD, HEAD_DIM, HIDDEN_DIM // HEAD]
        ).sum(dim=0)  # [HIDDEN_DIM // HEAD]

    if o_proj_module.bias is not None:
        assert isinstance(o_proj_module.bias, torch.Tensor)
        if bias is not None:
            bias = bias + o_proj_module.bias
        else:
            bias = o_proj_module.bias

    return ov_combined_weight_by_head, bias


def reconstruct_attn_output_vo_combined(
    attn_weights: Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE],
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    v_proj_module: torch.nn.Module,
    o_proj_module: torch.nn.Linear | Conv1D,
    num_attention_heads: int,
    num_kv_heads: int,
):

    assert isinstance(v_proj_module, torch.nn.Linear), (
        "Currently only supports linear v_proj modules."
    )
    head_dim = v_proj_module.out_features // num_kv_heads
    precomputed_ov_weights, precomputed_ov_bias = _precompute_ov_weights(
        v_proj_module=v_proj_module,
        o_proj_module=o_proj_module,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
    )

    value = torch.einsum(
        "bid,hdo->bhio",
        hidden_states,  # [BATCH, SEQUENCE, HIDDEN_DIM]
        precomputed_ov_weights,  # [HEAD, HEAD_DIM, HIDDEN_DIM // HEAD]
    ).contiguous()  # [BATCH, HEAD, SEQUENCE, HIDDEN_DIM // HEAD]

    weighted_value = torch.einsum(
        "bhij,bhjo->bio",
        attn_weights,  # [BATCH, HEAD, SEQUENCE, SEQUENCE]
        value,  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]
    )

    if precomputed_ov_bias is not None:
        weighted_value = weighted_value + precomputed_ov_bias

    return weighted_value
