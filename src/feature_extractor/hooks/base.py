from __future__ import annotations

from collections.abc import Callable

from torch import nn
from torch.utils.hooks import RemovableHandle

from feature_extractor.models.architecture import (
    BaseModelArchitecture,
    HookRegistrationConfig,
)

HookCallback = Callable[[nn.Module, tuple[object, ...], dict[str, object], object], None]


def register_forward_capture_hook(
    module: nn.Module,
    callback: HookCallback,
    *,
    config: HookRegistrationConfig,
) -> RemovableHandle:
    """Register a forward hook with a consistent callback signature."""

    if config.with_kwargs:
        return module.register_forward_hook(callback, with_kwargs=True)

    def wrapper(hooked_module: nn.Module, args: tuple[object, ...], output: object) -> None:
        callback(hooked_module, args, {}, output)

    return module.register_forward_hook(wrapper)


class HookManager:
    """Base class for managing multiple hooks for a model component."""

    def __init__(
        self,
        model: nn.Module,
        architecture: BaseModelArchitecture | None = None,
    ) -> None:
        self._model = model
        self._architecture = architecture or BaseModelArchitecture()
        self._hooks: list[RemovableHandle] = []

    def install(self) -> None:
        """Install hooks (no-op by default)."""

    def reset(self) -> None:
        """Reset any cached hook state."""

    def remove(self) -> None:
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def validate_layer_count(self, actual_layer_count: int) -> None:
        """Validate hook count matches expected layers (no-op by default)."""
