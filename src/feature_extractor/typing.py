from typing import Annotated

import torch
from typing_extensions import Generic, TypeVarTuple

T = TypeVarTuple("T")


class Tensor(Generic[*T], torch.Tensor):
    pass


BATCH = Annotated[int, "batch_size"]
LAYER = Annotated[int, "layer"]
SEQUENCE = Annotated[int, "length"]
HEAD = Annotated[int, "head"]
KV_HEAD = Annotated[int, "kv_head"]
HIDDEN_DIM = Annotated[int, "hidden_dim"]
HIDDEN_DIM_3 = Annotated[int, "hidden_dim_3"]
HEAD_DIM = Annotated[int, "head_dim"]
HALF_HEAD_DIM = Annotated[int, "half_head_dim"]
MLP_DIM = Annotated[int, "mlp_dim"]
