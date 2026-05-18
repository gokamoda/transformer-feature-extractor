import torch
from transformers.pytorch_utils import Conv1D

from feature_extractor.typing import BATCH, HEAD, HEAD_DIM, HIDDEN_DIM, SEQUENCE, Tensor


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


def create_causal_mask_single(
    sequence_length: int,
    dtype: torch.dtype,
):
    h = torch.full((sequence_length, sequence_length), torch.finfo(dtype).min)
    mask = torch.triu(h, diagonal=1)
    return mask
