import torch

from feature_extractor.typing import Tensor


class SimplifiedRoPEV1:
    def __init__(self, inv_freq: Tensor, attention_scaling: float = 1.0):
        self.inv_freq = inv_freq.to(torch.float32)
        self.device = self.inv_freq.device
        self.attention_scaling = attention_scaling

    def create_position_embeddings(self, sequence_length: int):
        position_ids = torch.arange(sequence_length).unsqueeze(0).to(self.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos[0], sin[0]
