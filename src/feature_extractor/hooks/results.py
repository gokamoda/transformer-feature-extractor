from dataclasses import dataclass
from feature_extractor.typing import Tensor, HEAD, SEQUENCE, HEAD_DIM, MLP_DIM, HIDDEN_DIM

@dataclass
class AttentionFeatures:
<<<<<<< HEAD
    layer_index: int
    query: None | Tensor[HEAD, SEQUENCE, HEAD_DIM]
=======
    layer: int
    query: None | Tensor[HEAD, SEQUENCE, HEAD_DIM] 
>>>>>>> 81938d6 (added layer index field to result classes)
    key: None | Tensor[HEAD, SEQUENCE, HEAD_DIM] # gqa unfurled
    value: None | Tensor[HEAD, SEQUENCE, HEAD_DIM] # gqa unfurled
    qk_logits: None | Tensor[HEAD, SEQUENCE, SEQUENCE]
    attn_weights: None | Tensor[HEAD, SEQUENCE, SEQUENCE]


@dataclass
class MLPFeatures:
<<<<<<< HEAD
    layer_index: int
=======
    layer: int
>>>>>>> 81938d6 (added layer index field to result classes)
    activation: None | Tensor[SEQUENCE, MLP_DIM]


@dataclass
class NormFeatures:
<<<<<<< HEAD
    layer_index: int
=======
    layer: int
    position: str   
>>>>>>> 81938d6 (added layer index field to result classes)
    input: None | Tensor[SEQUENCE, HIDDEN_DIM]
    output: None | Tensor[SEQUENCE, HIDDEN_DIM]

@dataclass
class LayerFeatures:
<<<<<<< HEAD
    layer_index: int
=======
    layer: int
>>>>>>> 81938d6 (added layer index field to result classes)
    input: None | Tensor[SEQUENCE, HIDDEN_DIM]
    attn_output: None | Tensor[SEQUENCE, HIDDEN_DIM]
    mlp_output: None | Tensor[SEQUENCE, HIDDEN_DIM]
    output: None | Tensor[SEQUENCE, HIDDEN_DIM]

@dataclass
class ExtractorResult:
    embeddings: None | Tensor[SEQUENCE, HIDDEN_DIM]
    layer_features: list[LayerFeatures]
    attention_features: list[AttentionFeatures]
    mlp_features: list[MLPFeatures]

    

