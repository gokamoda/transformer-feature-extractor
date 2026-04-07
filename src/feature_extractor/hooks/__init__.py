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
