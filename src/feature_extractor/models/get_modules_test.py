from types import SimpleNamespace

import pytest
import torch

from feature_extractor.models import (
    get_absolute_pos_embedding_module,
    get_o_proj_module,
    get_word_embedding_module,
)
from feature_extractor.models.architecture import BaseModelArchitecture
from feature_extractor.models.gpt2 import GPT2Architecture


def test_get_embedding_modules_return_independent_copies():
    word_embeddings = torch.nn.Embedding(10, 4)
    position_embeddings = torch.nn.Embedding(20, 4)
    model = SimpleNamespace(
        transformer=SimpleNamespace(wte=word_embeddings, wpe=position_embeddings)
    )
    architecture = GPT2Architecture()

    word_copy = get_word_embedding_module(architecture, model=model)
    position_copy = get_absolute_pos_embedding_module(architecture, model=model)

    assert isinstance(word_copy, torch.nn.Embedding)
    assert isinstance(position_copy, torch.nn.Embedding)
    assert word_copy is not word_embeddings
    assert position_copy is not position_embeddings
    assert torch.equal(word_copy.weight, word_embeddings.weight)
    assert torch.equal(position_copy.weight, position_embeddings.weight)


def test_get_absolute_pos_embedding_module_requires_architecture_field():
    with pytest.raises(AssertionError, match="absolute_pos_embedding_field"):
        get_absolute_pos_embedding_module(
            BaseModelArchitecture(),
            model=SimpleNamespace(model=SimpleNamespace()),
        )


def test_get_o_proj_module_does_not_print_cuda_memory(capsys):
    output_projection = torch.nn.Linear(4, 4)
    model = SimpleNamespace(
        transformer=SimpleNamespace(
            h=[SimpleNamespace(attn=SimpleNamespace(c_proj=output_projection))]
        )
    )

    output_copy = get_o_proj_module(GPT2Architecture(), layer_index=0, model=model)

    assert output_copy is not output_projection
    assert capsys.readouterr().out == ""
