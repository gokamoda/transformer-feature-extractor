import math
from dataclasses import dataclass
from typing import Literal

import torch
from transformers.pytorch_utils import Conv1D

from feature_extractor.models.architecture import BaseModelArchitecture
from feature_extractor.reconstruction.attention_weights import (
    _apply_rope,
    apply_mask,
    create_causal_mask,
)
from feature_extractor.reconstruction.rope import SimplifiedRoPEV1
from feature_extractor.typing import (
    BATCH,
    HALF_HEAD_DIM,
    HEAD,
    HEAD_DIM,
    HIDDEN_DIM,
    SEQUENCE,
    Tensor,
)


def rope_frequency_inner_products(
    query: Tensor[SEQUENCE, HEAD_DIM],
    key: Tensor[SEQUENCE, HEAD_DIM],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    scale_by_head_dim: bool = True,
) -> Tensor[HALF_HEAD_DIM, SEQUENCE, SEQUENCE]:
    """Compute RoPE QK logits separately for each rotation frequency.

    The returned tensor has shape ``[head_dim // 2, sequence, sequence]``.
    Summing it over the first dimension reconstructs the full RoPE QK logits.
    ``position_embeddings`` must contain the cosine and sine tensors used by
    the model for this sequence and head.
    """
    if query.shape != key.shape:
        raise ValueError(
            f"query and key must have the same shape, got {query.shape} and {key.shape}"
        )
    if query.ndim != 2:
        raise ValueError(
            f"query/key must have shape [sequence, head_dim], got {query.shape}"
        )

    _, head_dim = query.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    cos, sin = position_embeddings
    if cos.shape != sin.shape:
        raise ValueError(
            f"RoPE cosine and sine shapes must match, got {cos.shape} and {sin.shape}"
        )

    query_roped, key_roped = _apply_rope(
        query[None, None], key[None, None], position_embeddings
    )
    if query_roped.shape[:2] != (1, 1):
        raise ValueError(
            "position_embeddings must describe a single batch and head when "
            "query/key have shape [sequence, head_dim]"
        )
    query_roped = query_roped[0, 0]
    key_roped = key_roped[0, 0]

    half_head_dim = head_dim // 2
    logits_by_frequency = (
        torch.einsum(
            "if,jf->fij",
            query_roped[:, :half_head_dim],
            key_roped[:, :half_head_dim],
        )
        + torch.einsum(
            "if,jf->fij",
            query_roped[:, half_head_dim:],
            key_roped[:, half_head_dim:],
        )
    )

    if scale_by_head_dim:
        logits_by_frequency = logits_by_frequency / math.sqrt(head_dim)
    return logits_by_frequency


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
        ).transpose(0, 1)
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
        ).transpose(0, 1)
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
    norm_module: torch.nn.Module | None = None,
) -> Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]:
    """Reconstruct per-head query/key/value vectors from hidden_states.

    `norm_module` is the optional per-head RMSNorm some architectures apply to
    q_proj/k_proj output before RoPE (e.g. Qwen3/Gemma3's q_norm/k_norm, see
    `BaseModelArchitecture.attn_q_norm_field`/`attn_k_norm_field`). It is
    applied over the head_dim axis, before the heads/sequence transpose (the
    two orders are equivalent since RMSNorm only touches the last axis).
    Pass it whenever the architecture defines one, or the reconstructed
    query/key will be missing that normalization and be numerically wrong.
    """

    assert isinstance(qkv_proj_module, torch.nn.Linear), (
        "Currently only supports linear qkv_proj modules."
    )

    projected = qkv_proj_module(hidden_states)

    if module_type == "q_proj":
        head_dim = qkv_proj_module.out_features // num_attention_heads
        projected = projected.view(
            projected.shape[0],
            projected.shape[1],
            num_attention_heads,
            head_dim,
        )
        if norm_module is not None:
            projected = norm_module(projected)
        return projected.transpose(1, 2)
    else:
        assert num_kv_heads is not None, (
            "num_kv_heads must be provided for k_proj and v_proj reconstruction"
        )
        head_dim = qkv_proj_module.out_features // num_kv_heads
        projected = projected.view(
            projected.shape[0],
            projected.shape[1],
            num_kv_heads,
            head_dim,
        )
        if norm_module is not None:
            projected = norm_module(projected)
        projected = projected.transpose(1, 2)
        return projected.repeat_interleave(num_attention_heads // num_kv_heads, dim=1)


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


@dataclass
class QKBiasTerms:
    """The additive terms `_precompute_qk_weights` can't fold into
    `qk_weight_combined`, from expanding score(i,j) = (W_q x_i + b_q) . (W_k x_j + b_k):

        score(i,j) = x_i^T W_q^T W_k x_j   [-> qk_weight_combined]
                   + b_q . W_k x_j          [-> key_side, varies with j only]
                   + x_i^T W_q^T . b_k      [-> query_side, varies with i only]
                   + b_q . b_k              [-> constant]

    Under RoPE, W_q/W_k are each first rotated by a position-dependent
    R_i/R_j, so every term above picks up an (i, j) dependence through the
    relative rotation M_ij = R_i^T R_j: key_side/query_side/constant each
    gain leading (HEAD, SEQUENCE, SEQUENCE, ...) dims instead of just (HEAD, ...).
    Any field is None when the corresponding bias (q_proj's, k_proj's, or
    both) doesn't exist.
    """

    key_side: Tensor[HEAD, HIDDEN_DIM] | Tensor[HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM] | None
    query_side: Tensor[HEAD, HIDDEN_DIM] | Tensor[HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM] | None
    constant: Tensor[HEAD] | Tensor[HEAD, SEQUENCE, SEQUENCE] | None


def _precompute_qk_weights(
    q_proj_module: torch.nn.Linear,
    k_proj_module: torch.nn.Linear,
    num_attention_heads: int,
    head_dim: int,
    num_kv_heads: int,
    rope_module: SimplifiedRoPEV1 | None = None,
    sequence_length: int | None = None,
    architecture: BaseModelArchitecture | None = None,
) -> tuple[
    Tensor[HEAD, HIDDEN_DIM, HEAD_DIM]
    | Tensor[HEAD, SEQUENCE, SEQUENCE, HEAD_DIM, HEAD_DIM],
    QKBiasTerms,
]:
    """Fold q_proj/k_proj into a single static weight tensor such that
    `hidden_states @ qk_weight_combined @ hidden_states^T` (optionally
    RoPE-rotated) reproduces the bilinear part of the raw attention scores.
    Any q_proj/k_proj bias can't be folded into that same bilinear weight
    (it would need a nonexistent hidden_dim feature that is always 1), so it
    is returned separately as `QKBiasTerms` -- see there for the derivation.
    """
    if architecture is not None and (
        architecture.attn_q_norm_field is not None
        or architecture.attn_k_norm_field is not None
    ):
        raise NotImplementedError(
            "_precompute_qk_weights folds q_proj/k_proj into a single static "
            "weight matrix, which assumes the path between them and the "
            "attention scores is linear. Architectures with attn_q_norm_field/"
            "attn_k_norm_field set apply a per-head RMSNorm (nonlinear) to "
            "query/key before RoPE, so that fusion cannot represent them "
            "correctly. Use reconstruct_qkv_vectors(norm_module=...) followed "
            "by reconstruct_attention_weights instead."
        )

    q_proj_by_head_weight, q_proj_by_head_bias = _split_q_proj_by_head(
        q_proj_module=q_proj_module,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
    )
    num_heads, hidden_dim, _ = q_proj_by_head_weight.shape

    k_proj_by_head_weight, k_proj_by_head_bias = _split_kv_proj_by_head(
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
        # e: head_dim (query side, pre-rotation)
        # i: sequence_length (query side)
        # j: sequence_length (key side)
        # f: head_dim (key side, pre-rotation)
        # k: hidden_dim (key side)
        rope_matrix = rope_matrix.to(q_proj_by_head_weight.device)
        k_proj_by_head_weight = k_proj_by_head_weight.to(q_proj_by_head_weight.device)
        qk_weight_combined = torch.empty(
            (num_heads, sequence_length, sequence_length, hidden_dim, hidden_dim),
            dtype=q_proj_by_head_weight.dtype,
            device=q_proj_by_head_weight.device,
        )
        key_side = (
            torch.empty(
                (num_heads, sequence_length, sequence_length, hidden_dim),
                dtype=q_proj_by_head_weight.dtype,
                device=q_proj_by_head_weight.device,
            )
            if q_proj_by_head_bias is not None
            else None
        )
        query_side = (
            torch.empty(
                (num_heads, sequence_length, sequence_length, hidden_dim),
                dtype=q_proj_by_head_weight.dtype,
                device=q_proj_by_head_weight.device,
            )
            if k_proj_by_head_bias is not None
            else None
        )
        constant = (
            torch.empty(
                (num_heads, sequence_length, sequence_length),
                dtype=q_proj_by_head_weight.dtype,
                device=q_proj_by_head_weight.device,
            )
            if q_proj_by_head_bias is not None and k_proj_by_head_bias is not None
            else None
        )
        for h in range(num_heads):
            qk_weight_combined[h] = torch.einsum(
                "qe,ijef,kf->ijqk",
                q_proj_by_head_weight[h],
                rope_matrix,
                k_proj_by_head_weight[h],
            )
            if key_side is not None:
                key_side[h] = torch.einsum(
                    "e,ijef,kf->ijk",
                    q_proj_by_head_bias[h],
                    rope_matrix,
                    k_proj_by_head_weight[h],
                )
            if query_side is not None:
                query_side[h] = torch.einsum(
                    "qe,ijef,f->ijq",
                    q_proj_by_head_weight[h],
                    rope_matrix,
                    k_proj_by_head_bias[h],
                )
            if constant is not None:
                constant[h] = torch.einsum(
                    "e,ijef,f->ij",
                    q_proj_by_head_bias[h],
                    rope_matrix,
                    k_proj_by_head_bias[h],
                )
    else:
        # h: head
        # q: hidden_dim (query side)
        # e: head_dim
        # k: hidden_dim (key side)
        qk_weight_combined = torch.einsum(
            "hqe,hke->hqk", q_proj_by_head_weight, k_proj_by_head_weight
        )
        key_side = (
            torch.einsum("he,hke->hk", q_proj_by_head_bias, k_proj_by_head_weight)
            if q_proj_by_head_bias is not None
            else None
        )
        query_side = (
            torch.einsum("hqe,he->hq", q_proj_by_head_weight, k_proj_by_head_bias)
            if k_proj_by_head_bias is not None
            else None
        )
        constant = (
            torch.einsum("he,he->h", q_proj_by_head_bias, k_proj_by_head_bias)
            if q_proj_by_head_bias is not None and k_proj_by_head_bias is not None
            else None
        )

    return qk_weight_combined, QKBiasTerms(
        key_side=key_side, query_side=query_side, constant=constant
    )


def _add_qk_bias_terms_norope(
    reconstructed_attn_scores: Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE],
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    qk_bias_terms: QKBiasTerms,
) -> Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]:
    if qk_bias_terms.key_side is not None:
        # varies with key position j only -> broadcast over query dim (2)
        reconstructed_attn_scores = reconstructed_attn_scores + torch.einsum(
            "hk,bjk->bhj", qk_bias_terms.key_side, hidden_states
        ).unsqueeze(2)
    if qk_bias_terms.query_side is not None:
        # varies with query position i only -> broadcast over key dim (3)
        reconstructed_attn_scores = reconstructed_attn_scores + torch.einsum(
            "hq,biq->bhi", qk_bias_terms.query_side, hidden_states
        ).unsqueeze(3)
    if qk_bias_terms.constant is not None:
        reconstructed_attn_scores = reconstructed_attn_scores + qk_bias_terms.constant.view(
            1, -1, 1, 1
        )
    return reconstructed_attn_scores


def _add_qk_bias_terms_rope(
    reconstructed_attn_scores: Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE],
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM],
    qk_bias_terms: QKBiasTerms,
) -> Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]:
    device = hidden_states.device
    if qk_bias_terms.key_side is not None:
        reconstructed_attn_scores = reconstructed_attn_scores + torch.einsum(
            "hijk,bjk->bhij", qk_bias_terms.key_side.to(device), hidden_states
        )
    if qk_bias_terms.query_side is not None:
        reconstructed_attn_scores = reconstructed_attn_scores + torch.einsum(
            "hijq,biq->bhij", qk_bias_terms.query_side.to(device), hidden_states
        )
    if qk_bias_terms.constant is not None:
        reconstructed_attn_scores = (
            reconstructed_attn_scores + qk_bias_terms.constant.to(device).unsqueeze(0)
        )
    return reconstructed_attn_scores


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

    attn_weights = attn_weights.to(value.device)
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
    qk_weight_combined, qk_bias_terms = _precompute_qk_weights(
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
    reconstructed_attn_scores = _add_qk_bias_terms_norope(
        reconstructed_attn_scores, hidden_states, qk_bias_terms
    )

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
    qk_weight_combined: Tensor[HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM, HIDDEN_DIM],
    qk_bias_terms: QKBiasTerms,
    head_dim: int,
):
    # b: batch
    # i: sequence_length (query side)
    # j: sequence_length (key side)
    # q: hidden_dim (query side)
    # k: hidden_dim (key side)
    qk_weight_combined = qk_weight_combined.to(hidden_states.device)
    reconstructed_attn_scores = torch.empty(
        (
            hidden_states.shape[0],  # batch
            qk_weight_combined.shape[0],  # head
            qk_weight_combined.shape[1],  # sequence_length (query side)
            qk_weight_combined.shape[2],  # sequence_length (key side)
        ),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # for loop for memory efficiency
    for b in range(hidden_states.shape[0]):  # batch
        for h in range(qk_weight_combined.shape[0]):  # head
            reconstructed_attn_scores[b, h] = torch.einsum(
                "iq,ijqk,jk->ij",
                hidden_states[b],  # [SEQUENCE, HIDDEN_DIM]
                qk_weight_combined[h],  # [SEQUENCE, SEQUENCE, HIDDEN_DIM, HIDDEN_DIM]
                hidden_states[b],  # [SEQUENCE, HIDDEN_DIM]
            )
    reconstructed_attn_scores = _add_qk_bias_terms_rope(
        reconstructed_attn_scores, hidden_states, qk_bias_terms
    )

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
