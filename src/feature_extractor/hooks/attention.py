from dataclasses import dataclass
from math import sqrt

import torch
from transformers import PreTrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from feature_extractor.configs import FeatureConfig
from feature_extractor.hooks.base import Hook
from feature_extractor.models import BaseModelArchitecture
from feature_extractor.models.architecture import (
    QKV_IMPLEMENTATION_CONV1D,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
    get_num_attn_heads,
    get_num_kv_heads,
    get_num_layers,
)
from feature_extractor.typing import (
    BATCH,
    HEAD,
    HEAD_DIM,
    HIDDEN_DIM,
    KV_HEAD,
    SEQUENCE,
    Tensor,
)

from .base import AbstractBatchResult


@dataclass(repr=False, init=False)
class LinearProjectionObservationResult(AbstractBatchResult):
    output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class LinearProjectionHook(Hook):
    result: LinearProjectionObservationResult

    def save_result(self, hook_result: dict):
        self.result = LinearProjectionObservationResult(**hook_result)


@dataclass(repr=False, init=False)
class AttnModuleObservationResult(AbstractBatchResult):
    position_embeddings: Tensor | tuple[Tensor, ...] | None
    attention_mask: Tensor
    output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    attn_weights: Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]


class AttentionHook(Hook):
    result: AttnModuleObservationResult

    def save_result(self, hook_result: dict):
        self.result = AttnModuleObservationResult(**hook_result)


@dataclass
class AttentionHookResult:
    query: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]
    key: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    value: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    attn_weights: None | Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]
    position_embeddings: None | Tensor | tuple[Tensor, ...]
    attention_mask: None | Tensor
    output: None | Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class AttentionHookManager:
    query_layer_indices: list[int]
    key_layer_indices: list[int]
    value_layer_indices: list[int]
    qkv_combined_layer_indices: list[int]
    attn_weights_layer_indices: list[int]
    position_embeddings_layer_indices: list[int]
    attention_mask_layer_indices: list[int]
    output_layer_indices: list[int]
    attn_weights_outputs_combined_layer_indices: list[int]
    query_hooks: list[LinearProjectionHook]
    key_hooks: list[LinearProjectionHook]
    value_hooks: list[LinearProjectionHook]
    qkv_combined_hooks: list[LinearProjectionHook]
    attn_module_hooks: list[AttentionHook]
    model_config: PreTrainedConfig

    def __init__(
        self,
        model: PreTrainedModel,
        architecture: BaseModelArchitecture,
        feature_cfg: FeatureConfig,
    ):
        self.model_architecture = architecture
        self.feature_cfg = feature_cfg
        self.model_config = model.config
        self.reset_hooks()
        self.layer_indices = self._resolve_layer_index(self.feature_cfg)
        self.check_layer_index_in_range(model)
        self.install_hooks(model)

    def reset_hooks(self):
        self.query_layer_indices = []
        self.key_layer_indices = []
        self.value_layer_indices = []
        self.qkv_combined_layer_indices = []
        self.query_hooks = []
        self.key_hooks = []
        self.value_hooks = []
        self.qkv_combined_hooks = []
        self.attn_module_hooks = []
        self.output_layer_indices = []
        self.attn_weights_layer_indices = []
        self.position_embeddings_layer_indices = []
        self.attention_mask_layer_indices = []
        self.attn_weights_outputs_combined_layer_indices = []

    def _resolve_layer_index(self, feature_cfg: FeatureConfig) -> list[int]:
        layer_indices = []
        for feature_name in feature_cfg.feature_names:
            if feature_name.startswith("attn."):
                parts = feature_name.split(".")
                if len(parts) == 3 and parts[1].startswith("layer_"):
                    try:
                        layer_index = int(parts[1].split("_")[1])
                        if parts[2] == "query":
                            self.query_layer_indices.append(layer_index)
                        elif parts[2] == "key":
                            self.key_layer_indices.append(layer_index)
                        elif parts[2] == "value":
                            self.value_layer_indices.append(layer_index)
                        elif parts[2] == "attn_weights":
                            self.attn_weights_layer_indices.append(layer_index)
                        elif parts[2] == "position_embeddings":
                            self.position_embeddings_layer_indices.append(layer_index)
                        elif parts[2] == "attention_mask":
                            self.attention_mask_layer_indices.append(layer_index)
                        elif parts[2] == "output":
                            self.output_layer_indices.append(layer_index)
                        else:
                            raise ValueError(
                                f"Invalid attention feature name: {feature_name}"
                            )
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid layer index in feature name: {feature_name}"
                        ) from e
        if self.model_architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
            qkv_combined_layer_indices = list(
                set(self.query_layer_indices)
                | set(self.key_layer_indices)
                | set(self.value_layer_indices)
            )
            qkv_combined_layer_indices.sort()
            self.qkv_combined_layer_indices = qkv_combined_layer_indices

        if (
            len(self.attn_weights_layer_indices) > 0
            or len(self.position_embeddings_layer_indices) > 0
            or len(self.attention_mask_layer_indices) > 0
            or len(self.output_layer_indices) > 0
        ):
            attn_weights_outputs_combined_layer_indices = list(
                set(self.attn_weights_layer_indices)
                | set(self.position_embeddings_layer_indices)
                | set(self.attention_mask_layer_indices)
                | set(self.output_layer_indices)
            )
            attn_weights_outputs_combined_layer_indices.sort()
            self.attn_weights_outputs_combined_layer_indices = (
                attn_weights_outputs_combined_layer_indices
            )

        return layer_indices

    def check_layer_index_in_range(self, model: PreTrainedModel):
        num_layers = get_num_layers(model.config, self.model_architecture)
        for layer_index in (
            self.query_layer_indices
            + self.key_layer_indices
            + self.value_layer_indices
            + self.qkv_combined_layer_indices
            + self.attn_weights_layer_indices
            + self.position_embeddings_layer_indices
            + self.attention_mask_layer_indices
            + self.output_layer_indices
        ):
            if layer_index < 0 or layer_index >= num_layers:
                raise ValueError(
                    f"Layer index {layer_index} is out of range for model with {num_layers} layers."
                )

    def install_hooks(self, model: PreTrainedModel):
        if (
            self.model_architecture.attn_qkv_implementation
            == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
        ):
            self._install_hooks_qkv_independent(model)
        else:
            self._install_hooks_qkv_combined(model)

        self._install_hooks_attn_module(model)

    def remove_hooks(self):
        for hook in self.query_hooks:
            hook.remove()
        for hook in self.key_hooks:
            hook.remove()
        for hook in self.value_hooks:
            hook.remove()
        for hook in self.qkv_combined_hooks:
            hook.remove()

    @staticmethod
    def need_attn_hook(feature_cfg: FeatureConfig) -> bool:
        for feature_name in feature_cfg.feature_names:
            if feature_name.startswith("attn."):
                return True
        return False

    def _install_hooks_qkv_independent(self, model: PreTrainedModel):
        layers_module = getattr(
            getattr(model, self.model_architecture.model_field),
            self.model_architecture.layers_field,
        )

        # query
        for index in self.query_layer_indices:
            attn_module = getattr(
                layers_module[index], self.model_architecture.attn_field
            )
            assert self.model_architecture.attn_q_proj_field is not None, (
                "attn_q_proj_field must be defined for independent linear QKV implementation"
            )
            self.query_hooks.append(
                LinearProjectionHook(
                    module=getattr(
                        attn_module, self.model_architecture.attn_q_proj_field
                    ),
                    to_cpu=True,
                    with_output=["output"],
                )
            )

        # key
        for index in self.key_layer_indices:
            attn_module = getattr(
                layers_module[index], self.model_architecture.attn_field
            )
            assert self.model_architecture.attn_k_proj_field is not None, (
                "attn_k_proj_field must be defined for independent linear QKV implementation"
            )
            self.key_hooks.append(
                LinearProjectionHook(
                    module=getattr(
                        attn_module, self.model_architecture.attn_k_proj_field
                    ),
                    to_cpu=True,
                    with_output=["output"],
                )
            )

        # value
        for index in self.value_layer_indices:
            attn_module = getattr(
                layers_module[index], self.model_architecture.attn_field
            )
            assert self.model_architecture.attn_v_proj_field is not None, (
                "attn_v_proj_field must be defined for independent linear QKV implementation"
            )
            self.value_hooks.append(
                LinearProjectionHook(
                    module=getattr(
                        attn_module, self.model_architecture.attn_v_proj_field
                    ),
                    to_cpu=True,
                    with_output=["output"],
                )
            )

    def _install_hooks_qkv_combined(self, model: PreTrainedModel):
        layers_module = getattr(
            getattr(model, self.model_architecture.model_field),
            self.model_architecture.layers_field,
        )

        for index in self.qkv_combined_layer_indices:
            attn_module = getattr(
                layers_module[index], self.model_architecture.attn_field
            )
            assert self.model_architecture.attn_qkv_proj_field is not None, (
                "attn_qkv_proj_field must be defined for conv1d QKV implementation"
            )
            self.qkv_combined_hooks.append(
                LinearProjectionHook(
                    module=getattr(
                        attn_module, self.model_architecture.attn_qkv_proj_field
                    ),
                    to_cpu=True,
                    with_output=["output"],
                )
            )

    def _install_hooks_attn_module(self, model: PreTrainedModel):
        layers_module = getattr(
            getattr(model, self.model_architecture.model_field),
            self.model_architecture.layers_field,
        )

        for layer_index in self.attn_weights_outputs_combined_layer_indices:
            attn_module = getattr(
                layers_module[layer_index], self.model_architecture.attn_field
            )
            self.attn_module_hooks.append(
                AttentionHook(
                    module=attn_module,
                    to_cpu=True,
                    with_args=self.model_architecture.attn_pos_args,
                    with_output=[
                        "output",
                        "attn_weights",
                    ],
                )
            )

    def get_features(self, num_layers: int) -> list[AttentionHookResult | None]:
        features = []
        query_features: list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None] = [
            None
        ] * num_layers
        key_features: list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None] = [None] * (
            num_layers
        )
        value_features: list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None] = [
            None
        ] * num_layers
        attn_weights_features: list[Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE] | None] = [
            None
        ] * num_layers
        position_embeddings_features: list[Tensor | tuple[Tensor, ...] | None] = [
            None
        ] * num_layers
        attention_mask_features: list[Tensor | None] = [None] * num_layers
        attn_output_features: list[Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | None] = [
            None
        ] * num_layers

        if (
            len(self.query_layer_indices) > 0
            or len(self.key_layer_indices) > 0
            or len(self.value_layer_indices) > 0
        ):
            if (
                self.model_architecture.attn_qkv_implementation
                == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
            ):
                query_features, key_features, value_features = (
                    self._get_features_qkv_independent(num_layers)
                )
            else:
                query_features, key_features, value_features = (
                    self._get_features_qkv_combined(num_layers)
                )

        if len(self.attn_weights_outputs_combined_layer_indices) > 0:
            (
                attn_weights_features,
                position_embeddings_features,
                attention_mask_features,
                attn_output_features,
            ) = (
                self._get_features_attn_module(num_layers)
            )

        for layer_index in self.attn_weights_layer_indices:
            if (
                query_features[layer_index] is not None
                and key_features[layer_index] is not None
            ):
                attn_weights_features[layer_index] = self._reconstruct_attn_weights(
                    query=query_features[layer_index],
                    key=key_features[layer_index],
                    attention_mask=attention_mask_features[layer_index],
                    position_embeddings=position_embeddings_features[layer_index],
                )

        for query, key, value, attn_weights, position_embeddings, attention_mask, output in zip(
            query_features,
            key_features,
            value_features,
            attn_weights_features,
            position_embeddings_features,
            attention_mask_features,
            attn_output_features,
        ):
            if (
                query is None
                and key is None
                and value is None
                and attn_weights is None
                and position_embeddings is None
                and attention_mask is None
                and output is None
            ):
                features.append(None)
            else:
                features.append(
                    AttentionHookResult(
                        query=query,
                        key=key,
                        value=value,
                        attn_weights=attn_weights,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        output=output,
                    )
                )

        return features

    def _reconstruct_attn_weights(
        self,
        query: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
        key: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM],
        attention_mask: Tensor | None,
        position_embeddings: Tensor | tuple[Tensor, ...] | None,
    ) -> Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]:
        if (
            self.model_architecture.attn_qkv_implementation
            == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
            and isinstance(position_embeddings, tuple)
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / sqrt(query.shape[-1])

        if self.model_architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
            q_len = query.shape[-2]
            k_len = key.shape[-2]
            causal_mask = torch.tril(
                torch.ones((q_len, k_len), device=attn_logits.device, dtype=torch.bool)
            ).view(1, 1, q_len, k_len)
            min_value = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~causal_mask, min_value)

        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask

        return torch.softmax(attn_logits.float(), dim=-1).to(query.dtype)

    def _get_features_qkv_independent(
        self, num_layers: int
    ) -> tuple[
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
    ]:
        query_features = [None] * num_layers
        key_features = [None] * num_layers
        value_features = [None] * num_layers

        num_attn_heads = get_num_attn_heads(self.model_config, self.model_architecture)
        num_kv_heads = get_num_kv_heads(self.model_config, self.model_architecture)
        for _layer_index in range(num_layers):
            query = None
            key = None
            value = None

            if _layer_index in self.query_layer_indices:
                query_hook = self.query_hooks[
                    self.query_layer_indices.index(_layer_index)
                ]
                query: Tensor[BATCH, SEQUENCE, HIDDEN_DIM] = query_hook.result.output
                query: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] = query.view(
                    query.shape[0],
                    query.shape[1],
                    num_attn_heads,
                    query.shape[2] // num_attn_heads,
                ).transpose(1, 2)  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]
                query_features[_layer_index] = query

            if _layer_index in self.key_layer_indices:
                key_hook = self.key_hooks[self.key_layer_indices.index(_layer_index)]
                key: Tensor[BATCH, SEQUENCE, HIDDEN_DIM] = key_hook.result.output
                key: Tensor[BATCH, KV_HEAD, SEQUENCE, HEAD_DIM] = key.view(
                    key.shape[0],
                    key.shape[1],
                    num_kv_heads,
                    key.shape[2] // num_kv_heads,
                ).transpose(1, 2)  # [BATCH, KV_HEAD, SEQUENCE, HEAD_DIM]
                key = repeat_kv(key, n_rep=num_attn_heads // num_kv_heads)
                key_features[_layer_index] = key

            if _layer_index in self.value_layer_indices:
                value_hook = self.value_hooks[
                    self.value_layer_indices.index(_layer_index)
                ]
                value: Tensor[BATCH, SEQUENCE, HIDDEN_DIM] = value_hook.result.output
                value: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] = value.view(
                    value.shape[0],
                    value.shape[1],
                    num_kv_heads,
                    value.shape[2] // num_kv_heads,
                ).transpose(1, 2)  # [BATCH, KV_HEAD, SEQUENCE, HEAD_DIM]
                value: Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] = repeat_kv(
                    value, n_rep=num_attn_heads // num_kv_heads
                )
                value_features[_layer_index] = value

        return query_features, key_features, value_features

    def _get_features_qkv_combined(
        self, num_layers: int
    ) -> tuple[
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
    ]:
        query_features = [None] * num_layers
        key_features = [None] * num_layers
        value_features = [None] * num_layers

        for _layer_index, hook in zip(
            self.qkv_combined_layer_indices, self.qkv_combined_hooks
        ):
            qkv_output = hook.result.output
            assert len(qkv_output.shape) == 3, (
                "Expected 3D tensor for combined QKV output"
            )  # (batch_size, seq_len, 3 * hidden_dim)
            hidden_dim = qkv_output.shape[-1] // 3

            query = None
            key = None
            value = None

            if _layer_index in self.query_layer_indices:
                query = qkv_output[:, :, :hidden_dim]
                num_attn_heads = get_num_attn_heads(
                    self.model_config, self.model_architecture
                )
                query = query.view(
                    query.shape[0],
                    query.shape[1],
                    num_attn_heads,
                    query.shape[2] // num_attn_heads,
                ).transpose(1, 2)  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]
                query_features[_layer_index] = query

            if _layer_index in self.key_layer_indices:
                key = qkv_output[:, :, hidden_dim : 2 * hidden_dim]
                key = key.view(
                    key.shape[0],
                    key.shape[1],
                    num_attn_heads,
                    key.shape[2] // num_attn_heads,
                ).transpose(1, 2)  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]
                key_features[_layer_index] = key

            if _layer_index in self.value_layer_indices:
                value = qkv_output[:, :, 2 * hidden_dim :]
                value = value.view(
                    value.shape[0],
                    value.shape[1],
                    num_attn_heads,
                    value.shape[2] // num_attn_heads,
                ).transpose(1, 2)  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]
                value_features[_layer_index] = value
        return query_features, key_features, value_features

    def _get_features_attn_module(
        self, num_layers: int
    ) -> tuple[
        list[Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE] | None],
        list[Tensor | tuple[Tensor, ...] | None],
        list[Tensor | None],
        list[Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | None],
    ]:
        attn_weights_features = [None] * num_layers
        position_embeddings_features = [None] * num_layers
        attention_mask_features = [None] * num_layers
        attn_output_features = [None] * num_layers

        for layer_index, hook in zip(
            self.attn_weights_outputs_combined_layer_indices,
            self.attn_module_hooks,
        ):
            if layer_index in self.attn_weights_layer_indices:
                attn_weights_features[layer_index] = hook.result.attn_weights
            if layer_index in self.position_embeddings_layer_indices:
                position_embeddings_features[layer_index] = (
                    hook.result.position_embeddings
                )
            if layer_index in self.attention_mask_layer_indices:
                attention_mask_features[layer_index] = hook.result.attention_mask
            if layer_index in self.output_layer_indices:
                attn_output_features[layer_index] = hook.result.output

        return (
            attn_weights_features,
            position_embeddings_features,
            attention_mask_features,
            attn_output_features,
        )

    def need_eager_attn(self) -> bool:
        return len(self.attn_weights_outputs_combined_layer_indices) > 0
