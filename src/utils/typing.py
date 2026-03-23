from typing import Annotated

import torch
from typing_extensions import Generic, TypeVarTuple

T = TypeVarTuple("T")


class Tensor(Generic[T], torch.Tensor):  # type: ignore
    pass


BATCH = Annotated[int, "batch_size"]
LAYER = Annotated[int, "layer"]
SEQUENCE = Annotated[int, "length"]
HEAD = Annotated[int, "head"]
HIDDEN_DIM = Annotated[int, "hidden_dim"]
HEAD_DIM = Annotated[int, "head_dim"]
