from dataclasses import dataclass

from .attention import AttentionHookManager, AttentionHookResult
from .embedding import EmbeddingHookManager, EmbeddingHookResult
from .layer import LayerHookManager, LayerHookResult
from .mlp import MLPHookManager, MLPHookResult

__all__ = [
    "EmbeddingHookManager",
    "EmbeddingHookResult",
    "LayerHookManager",
    "LayerHookResult",
    "AttentionHookResult",
    "AttentionHookManager",
    "MLPHookManager",
    "MLPHookResult",
]


@dataclass
class HookResult:
    embeddings: EmbeddingHookResult | None
    layers: list[LayerHookResult | None] | None
    attn: list[AttentionHookResult | None] | None
    mlp: list[MLPHookResult | None] | None

    def __repr__(self) -> str:
        msg = "HookResult(\n"
        if self.embeddings is not None:
            msg += "    embeddings=EmbeddingHookResult,\n"

        if self.layers is not None:
            assert isinstance(self.layers, list)
            msg += f"    layers=[{', '.join(['LayerHookResult' if layer is not None else 'None' for layer in self.layers])}],\n"

        if self.attn is not None:
            assert isinstance(self.attn, list)
            msg += f"    attn=[{', '.join(['AttentionHookResult' if attn is not None else 'None' for attn in self.attn])}],\n"
        if self.mlp is not None:
            assert isinstance(self.mlp, list)
            msg += f"    mlp=[{', '.join(['MLPHookResult' if mlp is not None else 'None' for mlp in self.mlp])}],\n"

        msg += ")"
        return msg
