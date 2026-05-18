import math
from typing import Literal

import torch
from transformers.pytorch_utils import Conv1D

from feature_extractor.reconstruction.attention_weights import (
    apply_mask,
    create_causal_mask,
)
from feature_extractor.reconstruction.rope import SimplifiedRoPEV1
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


def _split_kv_proj_by_head(
    kv_proj_module: torch.nn.Linear,
    head_dim: int,
    num_attention_heads: int,
    num_kv_heads: int,
) -> tuple[Tensor[HEAD, HIDDEN_DIM, HEAD_DIM], Tensor[HEAD, HEAD_DIM] | None]:
    n_repeat = num_attention_heads // num_kv_heads
    kv_proj_weight_by_head: Tensor[HEAD, HIDDEN_DIM, HEAD_DIM] = (
        kv_proj_module.weight.T.view(
            -1,  # output_dim // num_heads
            num_kv_heads,
            head_dim,
        )
        .transpose(0, 1)
        .contiguous()
    )  # [HEAD, HIDDEN_DIM, HEAD_DIM]
    kv_proj_weight_by_head = kv_proj_weight_by_head.repeat_interleave(n_repeat, dim=0)

    kv_proj_bias_by_head: Tensor[HEAD, HEAD_DIM] | None = None
    if kv_proj_module.bias is not None:
        kv_proj_bias_by_head = kv_proj_module.bias.view(
            num_kv_heads,
            head_dim,
        ).repeat_interleave(n_repeat, dim=0)

    return kv_proj_weight_by_head, kv_proj_bias_by_head


def _split_q_proj_by_head(
    q_proj_module: torch.nn.Linear,
    head_dim: int,
    num_attention_heads: int,
) -> tuple[Tensor[HEAD, HIDDEN_DIM, HEAD_DIM], Tensor[HEAD, HEAD_DIM] | None]:
    assert head_dim * num_attention_heads == q_proj_module.out_features, (
        f"Output dimension of q_proj ({q_proj_module.out_features}) must be equal to head_dim ({head_dim}) * num_attention_heads ({num_attention_heads})."
    )

    q_proj_weight_by_head: Tensor[HEAD, HIDDEN_DIM, HEAD_DIM] = (
        q_proj_module.weight.T.view(
            -1,  # output_dim // num_heads
            num_attention_heads,
            head_dim,
        )
        .transpose(0, 1)
        .contiguous()
    )  # [HEAD, HIDDEN_DIM, HEAD_DIM]

    q_proj_bias_by_head: Tensor[HEAD, HEAD_DIM] | None = None
    if q_proj_module.bias is not None:
        q_proj_bias_by_head = q_proj_module.bias.view(
            num_attention_heads,
            head_dim,
        )

    return q_proj_weight_by_head, q_proj_bias_by_head


def reconstruct_qkv_vectors(
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    qkv_proj_module: torch.nn.Module,
    num_attention_heads: int,
    module_type: Literal["q_proj", "k_proj", "v_proj"],
    num_kv_heads: int | None = None,
) -> Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]:

    assert isinstance(qkv_proj_module, torch.nn.Linear), (
        "Currently only supports linear qkv_proj modules."
    )

    if module_type == "q_proj":
        head_dim = qkv_proj_module.out_features // num_attention_heads
        proj_weight_by_head, proj_bias_by_head = _split_q_proj_by_head(
            q_proj_module=qkv_proj_module,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
        )
    else:
        assert num_kv_heads is not None, (
            "num_kv_heads must be provided for k_proj and v_proj reconstruction"
        )
        head_dim = qkv_proj_module.out_features // num_kv_heads
        proj_weight_by_head, proj_bias_by_head = _split_kv_proj_by_head(
            kv_proj_module=qkv_proj_module,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
        )

    value_vectors: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] = torch.einsum(
        "bid,hde->bihe",
        hidden_states,  # [BATCH, SEQUENCE, HIDDEN_DIM]
        proj_weight_by_head,  # [HEAD, HEAD_DIM, HIDDEN_DIM // HEAD]
    ).contiguous()  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]

    if proj_bias_by_head is not None:
        value_vectors = value_vectors + proj_bias_by_head

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
    v_proj_weight_by_head, v_proj_bias_by_head = _split_kv_proj_by_head(
        kv_proj_module=v_proj_module,
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


def _precompute_qk_weights(
    q_proj_module: torch.nn.Linear,
    k_proj_module: torch.nn.Linear,
    num_attention_heads: int,
    head_dim: int,
    num_kv_heads: int,
    rope_module: SimplifiedRoPEV1 | None = None,
    sequence_length: int | None = None,
) -> tuple[
    Tensor[HEAD, HIDDEN_DIM, HEAD_DIM]
    | Tensor[HEAD, SEQUENCE, SEQUENCE, HEAD_DIM, HEAD_DIM],
    Tensor[HEAD, HEAD_DIM] | Tensor | None,
]:

    q_proj_by_head_weight, q_proj_by_head_bias = _split_q_proj_by_head(
        q_proj_module=q_proj_module,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
    )

    k_proj_by_head_weight, _ = _split_kv_proj_by_head(
        kv_proj_module=k_proj_module,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
    )

    if rope_module is not None:
        assert sequence_length is not None
        rope_matrix: Tensor[SEQUENCE, SEQUENCE, HEAD_DIM, HEAD_DIM] = (
            rope_module.create_rope_matrix_full_sequence(
                sequence_length=sequence_length
            ).to(q_proj_by_head_weight.dtype)
        )

        # h: head
        # q: hidden_dim (query side)
        # e: head_dim
        # i: sequence_length (query side)
        # j: sequence_length (key side)
        # f: head_dim (key side)
        # k: hidden_dim (key side)
        qk_weight_combined = torch.einsum(
            "hqe,ijef,hkf->hijqk",
            q_proj_by_head_weight,
            rope_matrix,
            k_proj_by_head_weight,
        )

        if q_proj_by_head_bias is not None:
            raise NotImplementedError(
                "Bias combination for RoPE is not implemented yet."
            )
        else:
            qk_bias_combined = None
    else:
        # h: head
        # q: hidden_dim (query side)
        # e: head_dim
        # k: hidden_dim (key side)
        qk_weight_combined = torch.einsum(
            "hqe,hke->hqk", q_proj_by_head_weight, k_proj_by_head_weight
        )

        # h: head
        # e: head_dim
        # k: hidden_dim (key side)
        if q_proj_by_head_bias is not None:
            qk_bias_combined = torch.einsum(
                "he,hke->hk", q_proj_by_head_bias, k_proj_by_head_weight
            )
        else:
            qk_bias_combined = None

    return qk_weight_combined, qk_bias_combined


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


def reconstruct_attn_weight_qk_combined_norope(
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    q_proj_module: torch.nn.Linear,
    k_proj_module: torch.nn.Linear,
    num_attention_heads: int,
    head_dim: int,
    num_kv_heads: int,
):
    qk_weight_combined, qk_bias_combined = _precompute_qk_weights(
        q_proj_module=q_proj_module,
        k_proj_module=k_proj_module,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
    )
    # b: batch
    # i: sequence_length (query side)
    # j: sequence_length (key side)
    # q: hidden_dim (query side)
    # k: hidden_dim (key side)
    reconstructed_attn_scores = torch.einsum(
        "biq,hqk,bjk->bhij", hidden_states, qk_weight_combined, hidden_states
    )
    if qk_bias_combined is not None:
        reconstructed_attn_scores = reconstructed_attn_scores + torch.einsum(
            "hk,bjk->bhj", qk_bias_combined, hidden_states
        ).unsqueeze(2)

    reconstructed_attn_scores = reconstructed_attn_scores / math.sqrt(head_dim)

    mask = create_causal_mask(
        sequence_length=reconstructed_attn_scores.shape[-1],
    )
    masked_reconstructed_attn_scores = apply_mask(
        attn_weights=reconstructed_attn_scores,
        mask=mask,
        mask_value=torch.finfo(reconstructed_attn_scores.dtype).min,
    )

    attn_weights = torch.softmax(masked_reconstructed_attn_scores, dim=-1)

    return attn_weights


def reconstruct_attn_weight_qk_combined_with_rope(
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    q_proj_module: torch.nn.Linear,
    k_proj_module: torch.nn.Linear,
    rope_module: SimplifiedRoPEV1,
    num_attention_heads: int,
    head_dim: int,
    num_kv_heads: int,
):
    qk_weight_combined: Tensor[HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM, HIDDEN_DIM]
    qk_bias_combined: Tensor[HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM] | None
    qk_weight_combined, qk_bias_combined = _precompute_qk_weights(
        q_proj_module=q_proj_module,
        k_proj_module=k_proj_module,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        rope_module=rope_module,
        sequence_length=hidden_states.shape[1],
    )
    # b: batch
    # i: sequence_length (query side)
    # j: sequence_length (key side)
    # q: hidden_dim (query side)
    # k: hidden_dim (key side)
    reconstructed_attn_scores = torch.einsum(
        "biq,hijqk,bjk->bhij", hidden_states, qk_weight_combined, hidden_states
    )
    if qk_bias_combined is not None:
        reconstructed_attn_scores = reconstructed_attn_scores + torch.einsum(
            "hk,bjk->bhj", qk_bias_combined, hidden_states
        ).unsqueeze(2)

    reconstructed_attn_scores = reconstructed_attn_scores / math.sqrt(head_dim)

    mask = create_causal_mask(
        sequence_length=reconstructed_attn_scores.shape[-1],
    )
    masked_reconstructed_attn_scores = apply_mask(
        attn_weights=reconstructed_attn_scores,
        mask=mask,
        mask_value=torch.finfo(reconstructed_attn_scores.dtype).min,
    )

    attn_weights = torch.softmax(masked_reconstructed_attn_scores, dim=-1)

    return attn_weights
