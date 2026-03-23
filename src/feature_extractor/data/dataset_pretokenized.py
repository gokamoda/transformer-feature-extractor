from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

@dataclass
class PreTokenizedEntry:
    idx: str | int
    text: str | None
    token_ids: list[int]


class PreTokenizedTextDataset(Dataset):
    """PyTorch Dataset for pre-tokenized data rows."""

    entries: list[PreTokenizedEntry]

    def __init__(self, data: list[PreTokenizedEntry]) -> None:
        self.entries = data

    def __post_init__(self):
        # unique idx check
        idxs = [entry.idx for entry in self.entries]
        if len(set(idxs)) != len(idxs):
            raise ValueError("Duplicate idx values found in dataset")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> PreTokenizedEntry:
        return self.entries[idx]

def make_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(batch):
        idxs = [entry.idx for entry in batch]
        token_ids_list = [entry.token_ids for entry in batch]

        padded = tokenizer.pad(
            {"input_ids": token_ids_list},
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
