from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

@dataclass
class TextDataEntry:
    idx: str
    text: str

class TextDataset(Dataset):
    def __init__(self, data: list[TextDataEntry]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> TextDataEntry:
        return self.data[idx]
    
def create_collator(tokenizer: PreTrainedTokenizer):
    def collate_fn(batch: list[TextDataEntry]):
        texts = [entry.text for entry in batch]
        indices = [entry.idx for entry in batch]

        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "indices": indices,
        }

    return collate_fn