from dataclasses import dataclass

from .layer import LayerHookManager, LayerHookResult

__all__ = ["LayerHookManager"]


@dataclass
class HookResult:
    layers: list[LayerHookResult | None] | None
