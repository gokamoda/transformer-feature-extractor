from dataclasses import dataclass

from feature_extractor.typing import (
    HEAD,
    HEAD_DIM,
    HIDDEN_DIM,
    MLP_DIM,
    SEQUENCE,
    Tensor,
)


@dataclass
class AttentionFeatures:
    query: None | Tensor[HEAD, SEQUENCE, HEAD_DIM]
    key: None | Tensor[HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    value: None | Tensor[HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    qk_logits: None | Tensor[HEAD, SEQUENCE, SEQUENCE]
    attn_weights: None | Tensor[HEAD, SEQUENCE, SEQUENCE]


@dataclass
class MLPFeatures:
    activation: None | Tensor[SEQUENCE, MLP_DIM]


@dataclass
class NormFeatures:
    input: None | Tensor[SEQUENCE, HIDDEN_DIM]
    output: None | Tensor[SEQUENCE, HIDDEN_DIM]
