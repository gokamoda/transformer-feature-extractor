from dataclasses import dataclass

from .attention import AttentionHookManager, AttentionHookResult
from .layer import LayerHookManager, LayerHookResult
from .mlp import MLPHookManager, MLPHookResult

__all__ = [
    "LayerHookManager",
    "LayerHookResult",
    "AttentionHookResult",
    "AttentionHookManager",
    "MLPHookManager",
    "MLPHookResult",
]


@dataclass
class HookResult:
    layers: list[LayerHookResult | None] | None
    attn: list[AttentionHookResult | None] | None
    mlp: list[MLPHookResult | None] | None
