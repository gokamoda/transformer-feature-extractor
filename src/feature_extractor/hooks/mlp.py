from __future__ import annotations

import logging

import torch
from torch import nn

from feature_extractor.hooks.base import HookManager
from feature_extractor.models.architecture import (
    BaseModelArchitecture,
    MLP_IMPLEMENTATION_GATED,
)

_logger = logging.getLogger(__name__)


class MLPActivationCache:
    """Caches per-layer MLP activation outputs captured by hooks.

    Parameters
    ----------
    mlp_modules : list[nn.Module]
        MLP modules to hook for activation capture.
    activation_fn : callable
        Callable that computes an activation tensor from hook inputs with signature
        ``(module, inputs, output) -> torch.Tensor | None``.
    """

    def __init__(
        self,
        mlp_modules: list[nn.Module],
        activation_fn,
    ) -> None:
        self.activation_outputs: list[torch.Tensor | None] = [None] * len(mlp_modules)
        self._hooks = []
        for idx, module in enumerate(mlp_modules):
            self._hooks.append(
                module.register_forward_hook(
                    self._make_store_hook(self.activation_outputs, idx, activation_fn)
                )
            )

    @staticmethod
    def _make_store_hook(
        storage: list[torch.Tensor | None],
        index: int,
        activation_fn,
    ):
        def activation_hook(module, inputs, output):
            activation = activation_fn(module, inputs, output)
            if activation is None:
                storage[index] = None
                return
            if not isinstance(activation, torch.Tensor):
                msg = (
                    "MLP activation hook expected a Tensor output "
                    f"but received {type(activation)}."
                )
                raise TypeError(msg)
            storage[index] = activation.detach()

        return activation_hook

    def reset(self) -> None:
        for idx in range(len(self.activation_outputs)):
            self.activation_outputs[idx] = None

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()


class MLPHookManager(HookManager):
    """Hook manager for MLP activations."""

    def __init__(
        self,
        model: nn.Module,
        architecture: BaseModelArchitecture | None = None,
    ) -> None:
        super().__init__(model, architecture)
        self.activation_cache: MLPActivationCache | None = None
        self._warned_layer_fallback = False
        self._warned_mlp_fallback = False
        self._warned_activation_fallback = False

    def install(self) -> None:
        mlp_modules = self._resolve_mlp_modules()
        if not mlp_modules:
            msg = "Model does not expose MLP modules for activation capture."
            raise ValueError(msg)
        self.activation_cache = MLPActivationCache(mlp_modules, self._compute_activation)

    def reset(self) -> None:
        if self.activation_cache is not None:
            self.activation_cache.reset()

    def remove(self) -> None:
        if self.activation_cache is not None:
            self.activation_cache.remove()

    def validate_layer_count(self, actual_layer_count: int) -> None:
        if self.activation_cache is None:
            return
        actual = len(self.activation_cache.activation_outputs)
        if actual != actual_layer_count:
            msg = (
                "MLP activation hooks do not match model layer count. "
                f"Expected {actual_layer_count} layers but found {actual}."
            )
            raise ValueError(msg)

    def activation(self, layer_idx: int, sample_index: int) -> torch.Tensor:
        if self.activation_cache is None:
            msg = "MLP activation hooks are not installed."
            raise ValueError(msg)
        activation = self.activation_cache.activation_outputs[layer_idx]
        if activation is None:
            msg = f"Missing MLP activation output for layer {layer_idx}."
            raise ValueError(msg)
        if activation.dim() == 2:
            return activation.detach().cpu()
        if activation.dim() != 3:
            msg = (
                "MLP activation must be a 2D or 3D tensor "
                f"(got shape {tuple(activation.shape)})."
            )
            raise ValueError(msg)
        return activation[sample_index].detach().cpu()

    def _resolve_mlp_modules(self) -> list[nn.Module]:
        model = self._model
        architecture = self._architecture
        model_root = getattr(model, architecture.model_field, model)
        layers = getattr(model_root, architecture.layer_field, None)
        if layers is None:
            if not self._warned_layer_fallback:
                self._warn_architecture_fallback(
                    "layers",
                    f"{architecture.model_field}.{architecture.layer_field}",
                    "model.layers / layers / transformer.h",
                )
                self._warned_layer_fallback = True
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layers = model.model.layers
            elif hasattr(model, "layers"):
                layers = model.layers
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layers = model.transformer.h
        if layers is None:
            return []
        mlp_modules: list[nn.Module] = []
        for layer in layers:
            mlp = getattr(layer, architecture.mlp_field, None)
            if mlp is None:
                if not self._warned_mlp_fallback:
                    self._warn_architecture_fallback(
                        "mlp",
                        architecture.mlp_field,
                        "mlp / feed_forward / ffn",
                    )
                    self._warned_mlp_fallback = True
                mlp = (
                    getattr(layer, "mlp", None)
                    or getattr(layer, "feed_forward", None)
                    or getattr(layer, "ffn", None)
                )
            if mlp is None:
                continue
            mlp_modules.append(mlp)
        return mlp_modules

    def _compute_activation(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor | tuple | None,
    ) -> torch.Tensor | None:
        """Compute the activation tensor for a hooked MLP module.

        Parameters
        ----------
        module : nn.Module
            MLP module being hooked.
        inputs : tuple
            Inputs passed into the MLP module.
        output : torch.Tensor | tuple | None
            Output produced by the MLP module.

        Returns
        -------
        torch.Tensor | None
            Activation tensor when derivable, otherwise a fallback output or None.
        """
        if len(inputs) == 0:
            return self._fallback_activation(output)
        hidden_states = inputs[0]
        if not isinstance(hidden_states, torch.Tensor):
            return self._fallback_activation(output)
        if self._architecture.mlp_implementation == MLP_IMPLEMENTATION_GATED:
            activation = self._compute_gated_activation(module, hidden_states)
            if activation is not None:
                return activation
        activation = self._compute_standard_activation(module, hidden_states)
        if activation is not None:
            return activation
        return self._fallback_activation(output)

    def _compute_gated_activation(
        self, module: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute gated activations for MLPs with gate/up projections.

        Expects ``gate_proj`` and ``up_proj`` attributes to be present on the module.
        Returns ``act_fn(gate_proj(x)) * up_proj(x)`` when available.
        """
        if not (hasattr(module, "gate_proj") and hasattr(module, "up_proj")):
            return None
        gate = module.gate_proj(hidden_states)
        up = module.up_proj(hidden_states)
        act_fn = getattr(module, "act_fn", None)
        if act_fn is None:
            act_fn = getattr(module, "activation_fn", None)
        if act_fn is None:
            return gate * up
        if callable(act_fn):
            return act_fn(gate) * up
        return None

    def _compute_standard_activation(
        self, module: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute standard MLP activations using the first projection.

        Looks for common projection attributes (fc1, c_fc, w1, up_proj) and applies
        the module activation function when available.
        """
        proj = None
        for attr_name in ("fc1", "c_fc", "w1", "up_proj"):
            if hasattr(module, attr_name):
                proj = getattr(module, attr_name)(hidden_states)
                break
        if proj is None:
            return None
        act_fn = getattr(module, "act_fn", None)
        if act_fn is None:
            act_fn = getattr(module, "activation_fn", None)
        if act_fn is None:
            act_fn = getattr(module, "act", None)
        if act_fn is None:
            return proj
        if callable(act_fn):
            return act_fn(proj)
        return None

    def _fallback_activation(
        self, output: torch.Tensor | tuple | None
    ) -> torch.Tensor | None:
        if not self._warned_activation_fallback:
            _logger.warning(
                "MLP activation could not be derived from module internals; "
                "falling back to MLP output."
            )
            self._warned_activation_fallback = True
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        return None

    @staticmethod
    def _warn_architecture_fallback(target: str, field: str, fallback: str) -> None:
        _logger.warning(
            "Model architecture config did not resolve %s via %s; falling back to %s. "
            "Please verify your BaseModelArchitecture settings if you are using a "
            "custom architecture.",
            target,
            field,
            fallback,
        )
