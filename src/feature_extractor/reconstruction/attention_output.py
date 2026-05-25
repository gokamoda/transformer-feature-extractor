from typing import Literal

import torch
from transformers.pytorch_utils import Conv1D

from feature_extractor.reconstruction.attention_dissection import _split_o_proj_by_head
from feature_extractor.typing import BATCH, HEAD, HEAD_DIM, HIDDEN_DIM, SEQUENCE, Tensor

NON_ROPE_MASKED_BIAS = (
    -10000.0
)  # Matches GPT-2 masked_bias in transformers.modeling_gpt2.


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

    device = o_proj_module.weight.device
    weighted_value = weighted_value.to(device)
    # prepare o_proj weight for reconstruction
    if unfurl in ["head_wise", "token_wise"]:
        o_proj_weight_by_head = _split_o_proj_by_head(
            o_proj_module=o_proj_module,
            head_dim=head_dim,
            num_attention_heads=num_heads,
        ).to(device)



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
