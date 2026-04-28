import pytest
import torch

from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
    load_causal_model,
)
from feature_extractor.reconstruction.rope import SimplifiedRoPEV1
from feature_extractor.typing import BATCH, HIDDEN_DIM, SEQUENCE, Tensor


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_rope_reconstruction(model_name):

    architecture = get_model_architecture(model_name)

    # model uses rope
    if not architecture.attn_use_rope:
        print(f"Model {model_name} does not use RoPE, skipping test")
        pytest.skip(f"Model {model_name} does not use RoPE, skipping test")
    else:
        assert architecture.rope_field is not None
        model_module = getattr(load_causal_model(model_name), architecture.model_field)
        original_rope_module = getattr(model_module, architecture.rope_field)

        hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM] = torch.randn(2, 3, 5)
        position_ids = torch.tensor([[0, 1, 2], [0, 0, 1]])
        cos_original, sin_original = original_rope_module(
            hidden_states, position_ids=position_ids
        )

        simplified_rope_moduel = SimplifiedRoPEV1(
            inv_freq=original_rope_module.inv_freq,
            attention_scaling=original_rope_module.attention_scaling,
        )
        for i in range(position_ids.shape[0]):
            sequence_length = position_ids[i, -1].item() + 1
            cos_simplified, sin_simplified = (
                simplified_rope_moduel.create_position_embeddings(
                    sequence_length=int(sequence_length)
                )
            )
            assert torch.allclose(cos_simplified, cos_original[i, -sequence_length:])
            assert torch.allclose(sin_simplified, sin_original[i, -sequence_length:])
