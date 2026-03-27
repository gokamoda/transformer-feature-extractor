from dataclasses import dataclass

from transformers import PreTrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import repeat_kv

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


@dataclass
class AttentionHookResult:
    query: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]
    key: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    value: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    attn_weights: None | Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]


class AttentionHookManager:
    query_layer_indices: list[int]
    key_layer_indices: list[int]
    value_layer_indices: list[int]
    qkv_combined_layer_indices: list[int]
    query_hooks: list[LinearProjectionHook]
    key_hooks: list[LinearProjectionHook]
    value_hooks: list[LinearProjectionHook]
    qkv_combined_hooks: list[LinearProjectionHook]
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
        self.install_hook(model)

    def reset_hooks(self):
        self.query_layer_indices = []
        self.key_layer_indices = []
        self.value_layer_indices = []
        self.qkv_combined_layer_indices = []
        self.query_hooks = []
        self.key_hooks = []
        self.value_hooks = []
        self.qkv_combined_hooks = []

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
        return layer_indices

    def check_layer_index_in_range(self, model: PreTrainedModel):
        num_layers = get_num_layers(model, self.model_architecture)
        for layer_index in (
            self.query_layer_indices
            + self.key_layer_indices
            + self.value_layer_indices
            + self.qkv_combined_layer_indices
        ):
            if layer_index < 0 or layer_index >= num_layers:
                raise ValueError(
                    f"Layer index {layer_index} is out of range for model with {num_layers} layers."
                )

    def install_hook(self, model: PreTrainedModel):
        if (
            self.model_architecture.attn_qkv_implementation
            == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
        ):
            self._install_hook_qkv_independent(model)
        else:
            self._install_hook_qkv_combined(model)

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

    def _install_hook_qkv_independent(self, model: PreTrainedModel):
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

    def _install_hook_qkv_combined(self, model: PreTrainedModel):
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

    def get_features(self, num_layers: int) -> list[AttentionHookResult | None]:
        [None] * num_layers

        if (
            self.model_architecture.attn_qkv_implementation
            == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
        ):
            return self._get_features_qkv_independent(num_layers)
        else:
            return self._get_features_qkv_combined(num_layers)

    def _get_features_qkv_independent(
        self, num_layers: int
    ) -> list[AttentionHookResult | None]:
        features = [None] * num_layers

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

            if query is not None or key is not None or value is not None:
                features[_layer_index] = AttentionHookResult(
                    query=query, key=key, value=value, attn_weights=None
                )

        return features

    def _get_features_qkv_combined(
        self, num_layers: int
    ) -> list[AttentionHookResult | None]:
        features = [None] * num_layers

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

            if _layer_index in self.key_layer_indices:
                key = qkv_output[:, :, hidden_dim : 2 * hidden_dim]
                key = key.view(
                    key.shape[0],
                    key.shape[1],
                    num_attn_heads,
                    key.shape[2] // num_attn_heads,
                ).transpose(1, 2)  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]

            if _layer_index in self.value_layer_indices:
                value = qkv_output[:, :, 2 * hidden_dim :]
                value = value.view(
                    value.shape[0],
                    value.shape[1],
                    num_attn_heads,
                    value.shape[2] // num_attn_heads,
                ).transpose(1, 2)  # [BATCH, HEAD, SEQUENCE, HEAD_DIM]

            features[_layer_index] = AttentionHookResult(
                query=query, key=key, value=value, attn_weights=None
            )

        return features
