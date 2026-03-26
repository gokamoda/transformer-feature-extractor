from .layer import LayerHookManager, LayerHookResult
from dataclasses import dataclass

__all__ = ["LayerHookManager"]



@dataclass
class HookResult:
    layers: list[LayerHookResult | None] | None