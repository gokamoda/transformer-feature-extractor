import torch

from feature_extractor.typing import HEAD_DIM, SEQUENCE, Tensor


class SimplifiedRoPEV1:
    def __init__(self, inv_freq: Tensor, attention_scaling: float = 1.0):
        self.inv_freq = inv_freq  # = theta
        self.dtype = inv_freq.dtype
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

    def create_rope_matrix_single(
        self, relative_position: int
    ) -> Tensor[HEAD_DIM, HEAD_DIM]:
        cos_matrix = torch.diag(
            torch.cos(self.inv_freq * relative_position).to(self.device)
        )
        sin_matrix = torch.diag(
            torch.sin(self.inv_freq * relative_position).to(self.device)
        )

        upper_half = torch.cat((cos_matrix, -sin_matrix), dim=-1)
        lower_half = torch.cat((sin_matrix, cos_matrix), dim=-1)
        rope_matrix = torch.cat((upper_half, lower_half), dim=0)
        return rope_matrix

    def create_rope_matrix_full_sequence(
        self, sequence_length: int
    ) -> Tensor[SEQUENCE, SEQUENCE, HEAD_DIM, HEAD_DIM]:

        relative_position_matrix = torch.arange(sequence_length).unsqueeze(
            0
        ) - torch.arange(sequence_length).unsqueeze(1)

        matrices = [
            [
                self.create_rope_matrix_single(
                    relative_position=int(relative_position_matrix[i, j])
                )
                for j in range(sequence_length)
            ]
            for i in range(sequence_length)
        ]

        return (
            torch.stack([torch.stack(row, dim=0) for row in matrices], dim=0)
            .to(self.device)
            .to(self.dtype)
        )
