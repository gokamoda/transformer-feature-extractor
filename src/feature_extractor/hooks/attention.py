from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from feature_extractor.hooks.base import HookManager
from feature_extractor.models.architecture import (
    BaseModelArchitecture,
    QKV_IMPLEMENTATION_CONV1D,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
)

_logger = logging.getLogger(__name__)


def _extract_attention_logits(module: nn.Module, output: object) -> torch.Tensor | None:
    for attr in ("attn_logits", "last_qk_logits", "attn_scores"):
        value = getattr(module, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(output, "attn_logits", None)
    if isinstance(value, torch.Tensor):
        return value
    value = getattr(output, "attn_scores", None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(output, dict):
        for key in ("attn_logits", "attn_scores"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    return None


def _extract_attention_weights(
    module: nn.Module, output: object
) -> torch.Tensor | None:
    for attr in ("attn_weights", "attn_probs", "attention_probs"):
        value = getattr(module, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(output, "attn_weights", None)
    if isinstance(value, torch.Tensor):
        return value
    value = getattr(output, "attn_probs", None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(output, dict):
        for key in ("attn_weights", "attn_probs", "attention_probs"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)) and len(output) > 1:
        # Some attention modules return post-softmax weights as the second element
        # in tuple outputs like (attn_output, attn_weights, ...), shaped
        # (batch, heads, seq_len, seq_len).
        if isinstance(output[1], torch.Tensor) and output[1].dim() == 4:
            return output[1]
    return None

@dataclass(frozen=True)
class AttentionHeadConfig:
    """Shape metadata for attention projections."""

    num_heads: int
    num_key_value_heads: int
    head_dim: int


class AttentionProjectionCache:
    """Caches per-layer attention projection outputs captured by hooks.

    Attributes
    ----------
    q_outputs, k_outputs, v_outputs
        Per-layer tensors captured from the projection modules.

    Methods
    -------
    reset()
        Clears cached tensors between forward passes.
    remove()
        Removes all registered hooks.
    validate_layer_count()
        Ensures hooks match the model's layer count.
    """
    def __init__(
        self,
        q_projections: list[nn.Module],
        k_projections: list[nn.Module],
        v_projections: list[nn.Module],
    ) -> None:
        """Register hooks for per-layer q/k/v projection outputs."""
        self.q_outputs: list[torch.Tensor | None] = [None] * len(q_projections)
        self.k_outputs: list[torch.Tensor | None] = [None] * len(k_projections)
        self.v_outputs: list[torch.Tensor | None] = [None] * len(v_projections)
        self._hooks = []
        for idx, module in enumerate(q_projections):
            self._hooks.append(
                module.register_forward_hook(
                    self._make_store_hook(self.q_outputs, idx)
                )
            )
        for idx, module in enumerate(k_projections):
            self._hooks.append(
                module.register_forward_hook(
                    self._make_store_hook(self.k_outputs, idx)
                )
            )
        for idx, module in enumerate(v_projections):
            self._hooks.append(
                module.register_forward_hook(
                    self._make_store_hook(self.v_outputs, idx)
                )
            )

    @staticmethod
    def _make_store_hook(storage: list[torch.Tensor | None], index: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                msg = (
                    "Attention projection hook expected a Tensor output "
                    f"but received {type(output)}."
                )
                raise TypeError(msg)
            storage[index] = output.detach()

        return hook

    def reset(self) -> None:
        """Clear cached projection outputs between batches."""
        for storage in (self.q_outputs, self.k_outputs, self.v_outputs):
            for idx in range(len(storage)):
                storage[idx] = None

    def remove(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()

    def validate_layer_count(self, expected: int) -> None:
        """Ensure the cached projections match the model layer count."""
        actual = len(self.q_outputs)
        if actual != expected:
            msg = (
                "Attention projection hooks do not match model layer count. "
                f"Expected {expected} layers but found {actual}."
            )
            raise ValueError(msg)


class CombinedAttentionProjectionCache:
    """Caches per-layer attention projection outputs from combined qkv modules."""

    def __init__(self, combined_modules: list[nn.Module]) -> None:
        self.q_outputs: list[torch.Tensor | None] = [None] * len(combined_modules)
        self.k_outputs: list[torch.Tensor | None] = [None] * len(combined_modules)
        self.v_outputs: list[torch.Tensor | None] = [None] * len(combined_modules)
        self._hooks = []
        for idx, module in enumerate(combined_modules):
            self._hooks.append(
                module.register_forward_hook(self._make_store_hook(idx))
            )

    def _make_store_hook(self, index: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                msg = (
                    "Combined attention projection hook expected a Tensor output "
                    f"but received {type(output)}."
                )
                raise TypeError(msg)
            if output.size(-1) % 3 != 0:
                msg = (
                    "Combined attention projection hidden size must be divisible by 3 "
                    f"(got {output.size(-1)})."
                )
                raise ValueError(msg)
            q, k, v = output.chunk(3, dim=-1)
            self.q_outputs[index] = q.detach()
            self.k_outputs[index] = k.detach()
            self.v_outputs[index] = v.detach()

        return hook

    def reset(self) -> None:
        for storage in (self.q_outputs, self.k_outputs, self.v_outputs):
            for idx in range(len(storage)):
                storage[idx] = None

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def validate_layer_count(self, expected: int) -> None:
        actual = len(self.q_outputs)
        if actual != expected:
            msg = (
                "Attention projection hooks do not match model layer count. "
                f"Expected {expected} layers but found {actual}."
            )
            raise ValueError(msg)


ProjectionCache = AttentionProjectionCache | CombinedAttentionProjectionCache


class AttentionOutputCache:
    """Caches per-layer attention outputs captured by hooks."""

    def __init__(self, attn_modules: list[nn.Module]) -> None:
        self.outputs: list[torch.Tensor | None] = [None] * len(attn_modules)
        self.qk_logits: list[torch.Tensor | None] = [None] * len(attn_modules)
        self.attn_weights: list[torch.Tensor | None] = [None] * len(attn_modules)
        self._hooks = []
        for idx, module in enumerate(attn_modules):
            self._hooks.append(
                module.register_forward_hook(
                    self._make_store_hook(
                        self.outputs, self.qk_logits, self.attn_weights, idx
                    )
                )
            )

    @staticmethod
    def _make_store_hook(
        storage: list[torch.Tensor | None],
        logits_storage: list[torch.Tensor | None],
        weights_storage: list[torch.Tensor | None],
        index: int,
    ):
        def hook(module, _inputs, output):
            output_tensor = None
            if isinstance(output, torch.Tensor):
                output_tensor = output
            elif isinstance(output, (tuple, list)) and output:
                if isinstance(output[0], torch.Tensor):
                    output_tensor = output[0]
            if output_tensor is None:
                msg = (
                    "Attention output hook expected a Tensor output "
                    f"but received {type(output)}."
                )
                raise TypeError(msg)
            storage[index] = output_tensor.detach()
            logits_tensor = _extract_attention_logits(module, output)
            if logits_tensor is not None:
                logits_storage[index] = logits_tensor.detach()
            weights_tensor = _extract_attention_weights(module, output)
            if weights_tensor is not None:
                weights_storage[index] = weights_tensor.detach()

        return hook

    def reset(self) -> None:
        for idx in range(len(self.outputs)):
            self.outputs[idx] = None
            self.qk_logits[idx] = None
            self.attn_weights[idx] = None

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def validate_layer_count(self, expected: int) -> None:
        actual = len(self.outputs)
        if actual != expected:
            msg = (
                "Attention output hooks do not match model layer count. "
                f"Expected {expected} layers but found {actual}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class QKVModuleGroup:
    q_modules: list[nn.Module]
    k_modules: list[nn.Module]
    v_modules: list[nn.Module]
    combined_modules: list[nn.Module]

    @property
    def has_independent(self) -> bool:
        return bool(self.q_modules)

    @property
    def has_combined(self) -> bool:
        return bool(self.combined_modules)


class AttentionHookManager(HookManager):
    """Manage attention hooks and reshape captured projections.

    Use ``install()`` once before inference, then call ``reset()`` per batch.
    Access per-layer projections via ``query()``, ``key()``, and ``value()``, and
    retrieve logits with ``qk_logits()`` when the attention module exposes them.
    Use ``remove()`` to clean up hooks.
    """
    def __init__(
        self,
        model: nn.Module,
        architecture: BaseModelArchitecture | None = None,
    ) -> None:
        """Initialize the manager for the provided model."""
        super().__init__(model, architecture)
        self.projection_cache: ProjectionCache | None = None
        self.attn_output_cache: AttentionOutputCache | None = None
        self.head_config: AttentionHeadConfig | None = None
        self._warned_layer_fallback = False
        self._warned_attention_fallback = False

    def install(
        self, *, required: bool = True, capture_attn_output: bool = False
    ) -> None:
        """Install projection hooks and resolve attention head metadata.

        Parameters
        ----------
        required : bool
            When True, raise if q/k/v projections cannot be resolved. When False,
            log a warning if hooks could not be installed.
        capture_attn_output : bool
            When True, capture per-layer attention outputs.

        """
        module_group = self._resolve_qkv_modules()
        cache_installed = self._init_projection_cache(module_group)
        if not cache_installed:
            if self._architecture.qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
                msg = "Model does not expose combined qkv projection modules."
            else:
                msg = "Model does not expose q/k/v projection modules."
            if required:
                raise ValueError(msg)
            _logger.warning("%s Falling back to model-provided attentions.", msg)

        if capture_attn_output:
            attn_modules = self._resolve_attention_modules()
            if attn_modules:
                self.attn_output_cache = AttentionOutputCache(attn_modules)
            else:
                _logger.warning(
                    "Model does not expose attention modules for output capture."
                )

    def reset(self) -> None:
        """Clear cached projections after each forward pass."""
        if self.projection_cache is not None:
            self.projection_cache.reset()
        if self.attn_output_cache is not None:
            self.attn_output_cache.reset()

    def remove(self) -> None:
        """Remove installed hooks."""
        if self.projection_cache is not None:
            self.projection_cache.remove()
        if self.attn_output_cache is not None:
            self.attn_output_cache.remove()

    def validate_layer_count(self, actual_layer_count: int) -> None:
        """Validate the number of hooks matches the model layer count."""
        if self.projection_cache is not None:
            self.projection_cache.validate_layer_count(actual_layer_count)
        if self.attn_output_cache is not None:
            self.attn_output_cache.validate_layer_count(actual_layer_count)

    def query(self, layer_idx: int, sample_index: int) -> torch.Tensor:
        """Return per-head query projection for the requested layer."""
        return self._prepare_attention_projection(
            self._projection_cache_or_raise().q_outputs,
            layer_idx,
            self._head_config_or_raise().num_heads,
            self._head_config_or_raise().head_dim,
            sample_index,
        )

    def key(self, layer_idx: int, sample_index: int) -> torch.Tensor:
        """Return per-head key projection for the requested layer."""
        head_config = self._head_config_or_raise()
        return self._prepare_attention_projection(
            self._projection_cache_or_raise().k_outputs,
            layer_idx,
            head_config.num_key_value_heads,
            head_config.head_dim,
            sample_index,
            num_attention_heads=head_config.num_heads,
        )

    def value(self, layer_idx: int, sample_index: int) -> torch.Tensor:
        """Return per-head value projection for the requested layer."""
        head_config = self._head_config_or_raise()
        return self._prepare_attention_projection(
            self._projection_cache_or_raise().v_outputs,
            layer_idx,
            head_config.num_key_value_heads,
            head_config.head_dim,
            sample_index,
            num_attention_heads=head_config.num_heads,
        )

    def qk_logits(self, layer_idx: int, sample_index: int) -> torch.Tensor | None:
        """Return attention logits captured from the attention module."""
        logits = self._attn_output_cache_or_raise().qk_logits[layer_idx]
        if logits is None:
            return None
        if logits.dim() != 4:
            msg = (
                "Attention logits must be a 4D tensor "
                f"(got shape {tuple(logits.shape)})."
            )
            raise ValueError(msg)
        return logits[sample_index].detach().cpu()

    def attn_weights(self, layer_idx: int, sample_index: int) -> torch.Tensor | None:
        """Return attention weights captured from the attention module."""
        weights = self._attn_output_cache_or_raise().attn_weights[layer_idx]
        if weights is None:
            return None
        if weights.dim() != 4:
            msg = (
                "Attention weights must be a 4D tensor "
                f"(got shape {tuple(weights.shape)})."
            )
            raise ValueError(msg)
        return weights[sample_index].detach().cpu()

    def attn_output(self, layer_idx: int, sample_index: int) -> torch.Tensor:
        """Return attention output for the requested layer."""
        output = self._attn_output_cache_or_raise().outputs[layer_idx]
        if output is None:
            msg = f"Missing attention output for layer {layer_idx}."
            raise ValueError(msg)
        if output.dim() != 3:
            msg = (
                "Attention output must be a 3D tensor "
                f"(got shape {tuple(output.shape)})."
            )
            raise ValueError(msg)
        return output[sample_index].detach().cpu()

    def _projection_cache_or_raise(self) -> ProjectionCache:
        if self.projection_cache is None:
            msg = "Attention projection hooks are not installed."
            raise ValueError(msg)
        return self.projection_cache

    def _head_config_or_raise(self) -> AttentionHeadConfig:
        if self.head_config is None:
            msg = "Attention head configuration is not available."
            raise ValueError(msg)
        return self.head_config

    def _attn_output_cache_or_raise(self) -> AttentionOutputCache:
        if self.attn_output_cache is None:
            msg = "Attention output hooks are not installed."
            raise ValueError(msg)
        return self.attn_output_cache

    def _init_projection_cache(self, module_group: QKVModuleGroup) -> bool:
        if self._architecture.qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
            if not module_group.has_combined:
                return False
            self.projection_cache = CombinedAttentionProjectionCache(
                module_group.combined_modules
            )
        elif self._architecture.qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
            if not module_group.has_independent:
                return False
            self.projection_cache = AttentionProjectionCache(
                module_group.q_modules,
                module_group.k_modules,
                module_group.v_modules,
            )
        else:
            return False
        self.head_config = self._resolve_attention_head_config()
        return True

    def _resolve_qkv_modules(self) -> QKVModuleGroup:
        layers = self._resolve_layers()
        if layers is None:
            return QKVModuleGroup([], [], [], [])

        q_modules: list[nn.Module] = []
        k_modules: list[nn.Module] = []
        v_modules: list[nn.Module] = []
        combined_modules: list[nn.Module] = []
        for attn in self._resolve_attention_modules(layers):
            if attn is None:
                continue
            if self._architecture.qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
                if hasattr(attn, "c_attn"):
                    combined_modules.append(attn.c_attn)
                continue
            if self._architecture.qkv_implementation == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR:
                if not all(
                    hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")
                ):
                    continue
                q_modules.append(attn.q_proj)
                k_modules.append(attn.k_proj)
                v_modules.append(attn.v_proj)
        return QKVModuleGroup(q_modules, k_modules, v_modules, combined_modules)

    def _resolve_layers(self) -> list[nn.Module] | None:
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
        return list(layers) if layers is not None else None

    def _resolve_attention_modules(
        self, layers: list[nn.Module] | None = None
    ) -> list[nn.Module]:
        if layers is None:
            layers = self._resolve_layers()
        if layers is None:
            return []
        architecture = self._architecture
        attn_modules: list[nn.Module] = []
        for layer in layers:
            attn = getattr(layer, architecture.attn_field, None)
            if attn is None:
                if not self._warned_attention_fallback:
                    self._warn_architecture_fallback(
                        "attention",
                        architecture.attn_field,
                        "self_attn / attn",
                    )
                    self._warned_attention_fallback = True
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                continue
            attn_modules.append(attn)
        return attn_modules

    @staticmethod
    def _warn_architecture_fallback(
        target: str,
        field: str,
        fallback: str,
    ) -> None:
        """Log when architecture config falls back to default attributes.

        Parameters
        ----------
        target : str
            Name of the module category being resolved (e.g., layers, attention).
        field : str
            Architecture field that failed to resolve.
        fallback : str
            Default attribute names used as the fallback.
        """
        _logger.warning(
            "Model architecture config did not resolve %s via %s; falling back to "
            "%s. Please verify your BaseModelArchitecture settings "
            "if you are using a custom architecture.",
            target,
            field,
            fallback,
        )

    def _resolve_attention_head_config(self) -> AttentionHeadConfig:
        config = getattr(self._model, "config", None)
        if config is None:
            msg = "Model config is required for attention head configuration."
            raise ValueError(msg)
        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(config, "n_head", None)
        if num_heads is None:
            msg = "Model config is missing num_attention_heads."
            raise ValueError(msg)
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(config, "n_embd", None)
            if hidden_size is None:
                msg = "Model config is missing hidden_size."
                raise ValueError(msg)
            head_dim = hidden_size // num_heads
        if num_heads % num_key_value_heads != 0:
            msg = (
                "num_attention_heads must be divisible by num_key_value_heads "
                f"(got {num_heads} and {num_key_value_heads})."
            )
            raise ValueError(msg)
        return AttentionHeadConfig(
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

    def _prepare_attention_projection(
        self,
        projections: list[torch.Tensor | None],
        layer_idx: int,
        projection_heads: int,
        head_dim: int,
        sample_index: int,
        num_attention_heads: int | None = None,
    ) -> torch.Tensor:
        projection = projections[layer_idx]
        if projection is None:
            msg = f"Missing attention projection output for layer {layer_idx}."
            raise ValueError(msg)
        if projection.dim() != 3:
            msg = (
                "Attention projection must be a 3D tensor "
                f"(got shape {tuple(projection.shape)})."
            )
            raise ValueError(msg)
        batch_size, seq_len, hidden_dim = projection.shape
        expected_hidden = projection_heads * head_dim
        if hidden_dim != expected_hidden:
            msg = (
                "Attention projection hidden size mismatch "
                f"(expected {expected_hidden}, got {hidden_dim})."
            )
            raise ValueError(msg)
        projection = projection.view(batch_size, seq_len, projection_heads, head_dim)
        # (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        projection = projection.transpose(1, 2)
        if num_attention_heads is not None and projection_heads != num_attention_heads:
            if num_attention_heads % projection_heads != 0:
                msg = (
                    "Cannot expand GQA heads "
                    f"(num_attention_heads={num_attention_heads}, "
                    f"num_key_value_heads={projection_heads})."
                )
                raise ValueError(msg)
            head_expansion_factor = num_attention_heads // projection_heads
            projection = projection.repeat_interleave(head_expansion_factor, dim=1)
        return projection[sample_index].detach().cpu()
