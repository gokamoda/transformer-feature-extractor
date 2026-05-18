import pytest
import torch

from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
    load_causal_model,
)
from feature_extractor.models.get_config import (
    get_hidden_size_per_head,
)
from feature_extractor.reconstruction.attention_weights import _apply_rope
from feature_extractor.reconstruction.rope import SimplifiedRoPEV1
from feature_extractor.typing import BATCH, HEAD, HIDDEN_DIM, SEQUENCE, Tensor


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


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_rope_matrix_reconstruction(model_name):

    architecture = get_model_architecture(model_name)

    # model uses rope
    if not architecture.attn_use_rope:
        print(f"Model {model_name} does not use RoPE, skipping test")
        pytest.skip(f"Model {model_name} does not use RoPE, skipping test")
    else:
        assert architecture.rope_field is not None
        model = load_causal_model(model_name)
        model_module = getattr(model, architecture.model_field)
        original_rope_module = getattr(model_module, architecture.rope_field)

        hidden_size = get_hidden_size_per_head(
            model_config=model.config,
            architecture=architecture,
        )
        head_size = get_hidden_size_per_head(
            model_config=model.config,
            architecture=architecture,
        )

        sequence_length = 5
        torch.randn(2, sequence_length, hidden_size)
        query: Tensor[BATCH, HEAD, SEQUENCE, HIDDEN_DIM] = torch.randn(
            2, 3, sequence_length, head_size
        )
        key: Tensor[BATCH, HEAD, SEQUENCE, HIDDEN_DIM] = torch.randn(
            2, 3, sequence_length, head_size
        )
        torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]])
        simplified_rope_module = SimplifiedRoPEV1(
            inv_freq=original_rope_module.inv_freq,
            attention_scaling=original_rope_module.attention_scaling,
        )
        position_embeddings = simplified_rope_module.create_position_embeddings(
            sequence_length=sequence_length
        )
        query_roped, key_roped = _apply_rope(query, key, position_embeddings)
        scores = torch.matmul(query_roped, key_roped.transpose(-2, -1))

        rope_matrix = simplified_rope_module.create_rope_matrix_full_sequence(
            sequence_length=sequence_length
        )
        naive_scores = torch.einsum(
            "bhqd,qkde,bhek->bhqk", query, rope_matrix, key.transpose(-2, -1)
        )

        torch.testing.assert_close(scores, naive_scores, rtol=1e-5, atol=1e-5)
