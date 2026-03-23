from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class Entry:
    idx: str | int
    text: str | None


class TextDataset(Dataset):
    """PyTorch Dataset for pre-tokenized data rows."""

    entries: list[Entry]

    def __init__(self, data: list[Entry]) -> None:
        self.entries = data

    def __post_init__(self):
        # unique idx check
        idxs = [entry.idx for entry in self.entries]
        if len(set(idxs)) != len(idxs):
            raise ValueError("Duplicate idx values found in dataset")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Entry:
        return self.entries[idx]

    @staticmethod
    def make_collate_fn(tokenizer: PreTrainedTokenizer):
        def collate_fn(batch):
            idxs = [entry.idx for entry in batch]
            texts = [entry.text for entry in batch]

            padded = tokenizer(
                texts,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            return {
                "idx": idxs,
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
            }

        return collate_fn
