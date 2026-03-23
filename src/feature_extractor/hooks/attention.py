from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from feature_extractor.hooks.base import HookManager
from feature_extractor.models.architecture import BaseModelArchitecture

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


class AttentionHookManager(HookManager):
    """Manage attention hooks and reshape captured projections.

    Use ``install()`` once before inference, then call ``reset()`` per batch.
    Access per-layer projections via ``query()``, ``key()``, and ``value()``, and
    compute logits with ``qk_logits()``. Use ``remove()`` to clean up hooks.
    """
    def __init__(
        self,
        model: nn.Module,
        architecture: BaseModelArchitecture | None = None,
    ) -> None:
        """Initialize the manager for the provided model."""
        super().__init__(model, architecture)
        self.projection_cache: AttentionProjectionCache | None = None
        self.head_config: AttentionHeadConfig | None = None

    def install(self) -> None:
        """Install projection hooks and resolve attention head metadata."""
        q_modules, k_modules, v_modules = self._resolve_qkv_modules()
        if not q_modules:
            msg = "Model does not expose q/k/v projection modules."
            raise ValueError(msg)
        self.projection_cache = AttentionProjectionCache(q_modules, k_modules, v_modules)
        self.head_config = self._resolve_attention_head_config()

    def reset(self) -> None:
        """Clear cached projections after each forward pass."""
        if self.projection_cache is not None:
            self.projection_cache.reset()

    def remove(self) -> None:
        """Remove installed hooks."""
        if self.projection_cache is not None:
            self.projection_cache.remove()

    def validate_layer_count(self, expected: int) -> None:
        """Validate the number of hooks matches the model layer count."""
        if self.projection_cache is not None:
            self.projection_cache.validate_layer_count(expected)

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

    def qk_logits(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention logits."""
        scaling_factor = 1.0 / math.sqrt(self._head_config_or_raise().head_dim)
        return torch.matmul(query, key.transpose(-2, -1)) * scaling_factor

    def _projection_cache_or_raise(self) -> AttentionProjectionCache:
        if self.projection_cache is None:
            msg = "Attention projection hooks are not installed."
            raise ValueError(msg)
        return self.projection_cache

    def _head_config_or_raise(self) -> AttentionHeadConfig:
        if self.head_config is None:
            msg = "Attention head configuration is not available."
            raise ValueError(msg)
        return self.head_config

    def _resolve_qkv_modules(
        self,
    ) -> tuple[list[nn.Module], list[nn.Module], list[nn.Module]]:
        model = self._model
        layers = None
        architecture = self._architecture
        model_root = getattr(model, architecture.model_field, model)
        layers = getattr(model_root, architecture.layer_field, None)
        if layers is None:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layers = model.model.layers
            elif hasattr(model, "layers"):
                layers = model.layers
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layers = model.transformer.h
        if layers is None:
            return ([], [], [])

        q_modules: list[nn.Module] = []
        k_modules: list[nn.Module] = []
        v_modules: list[nn.Module] = []
        for layer in layers:
            attn = getattr(layer, architecture.attn_field, None)
            if attn is None:
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                continue
            if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
                continue
            q_modules.append(attn.q_proj)
            k_modules.append(attn.k_proj)
            v_modules.append(attn.v_proj)
        return (q_modules, k_modules, v_modules)

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
