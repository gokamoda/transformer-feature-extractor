from dataclasses import dataclass

from transformers import PreTrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import repeat_kv

from feature_extractor.configs import FeatureConfig
from feature_extractor.configs.schema import AttentionFeatureSpec
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


class QKVHookManager:
    model_architecture: BaseModelArchitecture
    feature_cfg: FeatureConfig
    model_config: PreTrainedConfig

    query_layer_indices: list[int]
    key_layer_indices: list[int]
    value_layer_indices: list[int]

    qkv_combined_layer_indices: list[int]
    query_hooks: list[LinearProjectionHook]
    key_hooks: list[LinearProjectionHook]
    value_hooks: list[LinearProjectionHook]
    qkv_combined_hooks: list[LinearProjectionHook]

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
        self._resolve_layer_index(self.feature_cfg)
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

    def _resolve_layer_index(self, feature_cfg: FeatureConfig) -> None:
        for feature in feature_cfg.feature_specs:
            if not isinstance(feature, AttentionFeatureSpec):
                continue
            if feature.feature == "query":
                self.query_layer_indices.append(feature.layer_index)
            elif feature.feature == "key":
                self.key_layer_indices.append(feature.layer_index)
            elif feature.feature == "value":
                self.value_layer_indices.append(feature.layer_index)
        if self.model_architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D:
            qkv_combined_layer_indices = list(
                set(self.query_layer_indices)
                | set(self.key_layer_indices)
                | set(self.value_layer_indices)
            )
            qkv_combined_layer_indices.sort()
            self.qkv_combined_layer_indices = qkv_combined_layer_indices

    def check_layer_index_in_range(self, model: PreTrainedModel):
        num_layers = get_num_layers(model.config, self.model_architecture)
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

    def install_hooks(self, model: PreTrainedModel):
        if (
            self.model_architecture.attn_qkv_implementation
            == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
        ):
            self._install_hooks_qkv_independent(model)
        else:
            self._install_hooks_qkv_combined(model)

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

    def get_features(
        self,
    ) -> tuple[
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
        list[Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM] | None],
    ]:
        num_layers = get_num_layers(self.model_config, self.model_architecture)
        if (
            self.model_architecture.attn_qkv_implementation
            == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
        ):
            return self._get_features_qkv_independent(num_layers)
        else:
            return self._get_features_qkv_combined(num_layers)

    @staticmethod
    def need_qkv_hook(feature_cfg: FeatureConfig) -> bool:
        return any(
            isinstance(feature, AttentionFeatureSpec)
            and feature.feature in ["query", "key", "value"]
            for feature in feature_cfg.feature_specs
        )


@dataclass(repr=False, init=False)
class AttnModuleObservationResult(AbstractBatchResult):
    output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    attn_weights: Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]
    position_embeddings: tuple[Tensor, Tensor]
    attention_mask: Tensor


class AttentionHook(Hook):
    result: AttnModuleObservationResult

    def save_result(self, hook_result: dict):
        self.result = AttnModuleObservationResult(**hook_result)


class AttentionModuleHookManager:
    attn_weights_layer_indices: list[int]
    output_layer_indices: list[int]
    attention_mask_layer_indices: list[int]
    position_embeddings_layer_indices: list[int]
    layer_indices = list[int]

    attn_module_hooks: list[AttentionHook]

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
        self._resolve_layer_index(self.feature_cfg)
        self.check_layer_index_in_range(model)
        self.install_hooks(model)

    def reset_hooks(self):
        self.attn_weights_layer_indices = []
        self.output_layer_indices = []
        self.attn_weights_outputs_combined_layer_indices = []
        self.attn_module_hooks = []
        self.attention_mask_layer_indices = []
        self.position_embeddings_layer_indices = []

    def _resolve_layer_index(self, feature_cfg: FeatureConfig) -> None:
        for feature in feature_cfg.feature_specs:
            if not isinstance(feature, AttentionFeatureSpec):
                continue
            if feature.feature == "attn_weights":
                self.attn_weights_layer_indices.append(feature.layer_index)
            elif feature.feature == "output":
                self.output_layer_indices.append(feature.layer_index)
            elif feature.feature == "attention_mask":
                self.attention_mask_layer_indices.append(feature.layer_index)
            elif feature.feature == "positional_embedding":
                self.position_embeddings_layer_indices.append(feature.layer_index)

        if (
            len(self.attn_weights_layer_indices) > 0
            or len(self.output_layer_indices) > 0
            or len(self.attention_mask_layer_indices) > 0
            or len(self.position_embeddings_layer_indices) > 0
        ):
            combined_layer_indices = list(
                set(self.attn_weights_layer_indices)
                | set(self.output_layer_indices)
                | set(self.attention_mask_layer_indices)
                | set(self.position_embeddings_layer_indices)
            )
            combined_layer_indices.sort()
            self.layer_indices = combined_layer_indices

        else:
            self.layer_indices = []
            raise ValueError(
                "Something is wrong. check need_attn_module_hook function."
                "No valid attention module hook layer index found."
            )

    def check_layer_index_in_range(self, model: PreTrainedModel):
        num_layers = get_num_layers(model.config, self.model_architecture)

        assert isinstance(self.layer_indices, list), (
            "Expected layer_indices to be a list"
        )
        for layer_index in self.layer_indices:
            if layer_index < 0 or layer_index >= num_layers:
                raise ValueError(
                    f"Layer index {layer_index} is out of range for model with {num_layers} layers."
                )

    def install_hooks(self, model: PreTrainedModel):
        layers_module = getattr(
            getattr(model, self.model_architecture.model_field),
            self.model_architecture.layers_field,
        )

        assert isinstance(self.layer_indices, list), (
            "Expected layer_indices to be a list"
        )
        for layer_index in self.layer_indices:
            attn_module = getattr(
                layers_module[layer_index], self.model_architecture.attn_field
            )

            hook_kwargs = []
            if layer_index in self.attention_mask_layer_indices:
                if self.model_architecture.attn_attention_mask_arg_name is not None:
                    hook_kwargs.append(
                        self.model_architecture.attn_attention_mask_arg_name
                    )
            if layer_index in self.position_embeddings_layer_indices:
                if (
                    self.model_architecture.attn_position_embeddings_arg_name
                    is not None
                ):
                    hook_kwargs.append(
                        self.model_architecture.attn_position_embeddings_arg_name
                    )

            self.attn_module_hooks.append(
                AttentionHook(
                    module=attn_module,
                    to_cpu=True,
                    with_args=self.model_architecture.attn_pos_args,
                    with_kwargs=hook_kwargs,
                    with_output=[
                        "output",
                        "attn_weights",
                    ],
                )
            )

    def get_features(
        self,
    ) -> tuple[
        list[Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE] | None],
        list[Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | None],
        list[Tensor | tuple | None],
        list[Tensor | None],
    ]:
        num_layers = get_num_layers(self.model_config, self.model_architecture)
        attn_weights_features = [None] * num_layers
        attn_output_features = [None] * num_layers
        attention_mask_features = [None] * num_layers
        position_embedding_features = [None] * num_layers

        assert isinstance(self.layer_indices, list), (
            "Expected layer_indices to be a list"
        )
        for layer_index, hook in zip(self.layer_indices, self.attn_module_hooks):
            if layer_index in self.attn_weights_layer_indices:
                attn_weights_features[layer_index] = hook.result.attn_weights
            if layer_index in self.output_layer_indices:
                attn_output_features[layer_index] = hook.result.output
            if layer_index in self.attention_mask_layer_indices:
                attention_mask_features[layer_index] = hook.result.attention_mask
            if (
                layer_index in self.position_embeddings_layer_indices
                and self.model_architecture.attn_position_embeddings_arg_name
                is not None
            ):
                position_embedding_features[layer_index] = (
                    hook.result.position_embeddings
                )

        return (
            attn_weights_features,
            attn_output_features,
            attention_mask_features,
            position_embedding_features,
        )

    @staticmethod
    def need_attn_module_hook(feature_cfg: FeatureConfig) -> bool:
        return any(
            isinstance(feature, AttentionFeatureSpec)
            and feature.feature
            in [
                "attn_weights",
                "output",
                "attention_mask",
                "positional_embedding",
            ]
            for feature in feature_cfg.feature_specs
        )

    def need_eager_attn(self) -> bool:
        if not isinstance(self.layer_indices, list):
            return False
        return len(self.layer_indices) > 0


@dataclass
class AttentionHookResult:
    query: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]
    key: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    value: None | Tensor[BATCH, HEAD, SEQUENCE, HEAD_DIM]  # gqa unfurled
    attn_weights: None | Tensor[BATCH, HEAD, SEQUENCE, SEQUENCE]
    output: None | Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    position_embeddings: None | tuple[Tensor, Tensor]
    attention_mask: None | Tensor | tuple


class AttentionHookManager:
    qkv_hook_manager: QKVHookManager | None
    attn_module_hook_manager: AttentionModuleHookManager | None

    def __init__(
        self,
        model: PreTrainedModel,
        architecture: BaseModelArchitecture,
        feature_cfg: FeatureConfig,
    ):
        self.check_feature_cfg(feature_cfg)
        if QKVHookManager.need_qkv_hook(feature_cfg):
            self.qkv_hook_manager = QKVHookManager(
                model=model, architecture=architecture, feature_cfg=feature_cfg
            )
        else:
            self.qkv_hook_manager = None

        if AttentionModuleHookManager.need_attn_module_hook(feature_cfg):
            self.attn_module_hook_manager = AttentionModuleHookManager(
                model=model, architecture=architecture, feature_cfg=feature_cfg
            )
        else:
            self.attn_module_hook_manager = None

    def check_feature_cfg(self, feature_cfg: FeatureConfig):
        for feature in feature_cfg.feature_specs:
            if isinstance(feature, AttentionFeatureSpec):
                continue

    @staticmethod
    def need_attn_hook(feature_cfg: FeatureConfig) -> bool:
        return any(
            isinstance(feature, AttentionFeatureSpec)
            for feature in feature_cfg.feature_specs
        )

    def get_features(self, num_layers: int) -> list[AttentionHookResult | None]:
        features = []
        if self.qkv_hook_manager is not None:
            query_features, key_features, value_features = (
                self.qkv_hook_manager.get_features()
            )
        else:
            query_features, key_features, value_features = (
                [None] * num_layers,
                [None] * num_layers,
                [None] * num_layers,
            )

        if self.attn_module_hook_manager is not None:
            (
                attn_weights_features,
                attn_output_features,
                attention_mask_features,
                position_embedding_features,
            ) = self.attn_module_hook_manager.get_features()
        else:
            (
                attn_weights_features,
                attn_output_features,
                attention_mask_features,
                position_embedding_features,
            ) = (
                [None] * num_layers,
                [None] * num_layers,
                [None] * num_layers,
                [None] * num_layers,
            )

        for (
            query,
            key,
            value,
            attn_weights,
            output,
            attention_mask,
            position_embedding,
        ) in zip(
            query_features,
            key_features,
            value_features,
            attn_weights_features,
            attn_output_features,
            attention_mask_features,
            position_embedding_features,
        ):
            if (
                query is None
                and key is None
                and value is None
                and attn_weights is None
                and output is None
                and attention_mask is None
                and position_embedding is None
            ):
                features.append(None)
            else:
                features.append(
                    AttentionHookResult(
                        query=query,
                        key=key,
                        value=value,
                        attn_weights=attn_weights,
                        output=output,
                        attention_mask=attention_mask,
                        position_embeddings=position_embedding,
                    )
                )

        return features

    def need_eager_attn(self) -> bool:
        if self.attn_module_hook_manager is None:
            return False

        return self.attn_module_hook_manager.need_eager_attn()
