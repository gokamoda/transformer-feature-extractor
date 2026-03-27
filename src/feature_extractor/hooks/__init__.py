from dataclasses import dataclass

from .attention import AttentionHookManager, AttentionHookResult
from .layer import LayerHookManager, LayerHookResult

__all__ = [
    "LayerHookManager",
    "LayerHookResult",
    "AttentionHookResult",
    "AttentionHookManager",
]


@dataclass
class HookResult:
    layers: list[LayerHookResult | None] | None
    attn: list[AttentionHookResult | None] | None
