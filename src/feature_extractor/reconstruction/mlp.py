import torch
from transformers import PreTrainedModel

from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HIDDEN_DIM, MLP_DIM, SEQUENCE, Tensor


def reconstruct_mlp_output(
    down_proj_input: Tensor[BATCH, SEQUENCE, MLP_DIM],
    model: PreTrainedModel,
    layer_index: int,
    architecture: BaseModelArchitecture,
) -> Tensor[BATCH, SEQUENCE, HIDDEN_DIM]:
    """Reconstruct MLP output by applying the down projection to down_proj_input.

    This is equivalent to the MLP output for both standard and gated MLP
    implementations, since the down projection is the final linear layer in both
    cases.
    """
    layers_module = getattr(
        getattr(model, architecture.model_field),
        architecture.layers_field,
    )
    mlp_module = getattr(layers_module[layer_index], architecture.mlp_field)
    down_proj = getattr(mlp_module, architecture.mlp_down_proj_field)

    device = next(down_proj.parameters()).device
    dtype = next(down_proj.parameters()).dtype

    with torch.no_grad():
        return down_proj(down_proj_input.to(device=device, dtype=dtype)).cpu()
