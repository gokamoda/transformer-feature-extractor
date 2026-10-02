from unittest.mock import Mock

import torch

from feature_extractor.data.dataset import TextDataEntry, create_collator


def test_create_collator_applies_optional_max_length():
    tokenizer = Mock(
        return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )

    collate = create_collator(tokenizer, max_length=3)
    result = collate([TextDataEntry(idx="doc", text="a long document")])

    tokenizer.assert_called_once_with(
        ["a long document"],
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        truncation=True,
        max_length=3,
    )
    assert result["indices"] == ["doc"]


def test_create_collator_does_not_truncate_without_max_length():
    tokenizer = Mock(
        return_value={
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
    )

    create_collator(tokenizer)([TextDataEntry(idx="doc", text="text")])

    assert tokenizer.call_args.kwargs["truncation"] is False
    assert tokenizer.call_args.kwargs["max_length"] is None
