from dataclasses import dataclass

from transformers import PreTrainedModel

from feature_extractor.configs import FeatureConfig
from feature_extractor.hooks.base import Hook
from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HIDDEN_DIM, SEQUENCE, Tensor

from .base import AbstractBatchResult


@dataclass(repr=False, init=False)
class BatchLayerObservationResult(AbstractBatchResult):
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    hidden_states_output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class LayerHook(Hook):
    result: BatchLayerObservationResult

    def save_result(self, hook_result: dict):
        self.result = BatchLayerObservationResult(**hook_result)


@dataclass
class LayerHookResult:
    input: None | Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    output: None | Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class LayerHookManager:
    input_layer_indices: list[int]
    output_layer_indices: list[int]
    layer_indices: list[int]
    layer_hooks: list[LayerHook]

    def __init__(
        self,
        model: PreTrainedModel,
        architecture: BaseModelArchitecture,
        feature_cfg: FeatureConfig,
    ):
        self.model_architecture = architecture
        self.feature_cfg = feature_cfg
        self.reset_hooks()
        self._resolve_layer_index(self.feature_cfg)
        self.install_hook(model)

    def reset_hooks(self):
        self.input_layer_indices = []
        self.output_layer_indices = []
        self.layer_indices = []
        self.layer_hooks = []

    def install_hook(self, model: PreTrainedModel):
        layers_module = getattr(
            getattr(model, self.model_architecture.model_field),
            self.model_architecture.layers_field,
        )
        for index in self.layer_indices:
            hook_kwargs = {}
            if index in self.input_layer_indices:
                hook_kwargs['with_args'] = self.model_architecture.layers_pos_args
                hook_kwargs['with_kwargs'] = [self.model_architecture.layers_input_hidden_state_arg_name]
            if index in self.output_layer_indices:
                hook_kwargs['with_output'] = self.model_architecture.layer_return_fields

            self.layer_hooks.append(
                LayerHook(
                    module=layers_module[index],
                    to_cpu=True,
                    **hook_kwargs
                )
            )

    def remove_hooks(self):
        for hook in self.layer_hooks:
            hook.remove()

    @staticmethod
    def need_layer_hook(feature_cfg: FeatureConfig) -> bool:
        for feature_name in feature_cfg.feature_names:
            if feature_name.startswith("layers."):
                return True
        return False

    def _resolve_layer_index(self, feature_cfg: FeatureConfig) -> None:
        for feature_name in feature_cfg.feature_names:
            if feature_name.startswith("layers."):
                parts = feature_name.split(".")
                if len(parts) == 3 and parts[1].startswith("layer_"):
                    try:
                        layer_index = int(parts[1].split("_")[1])
                        if parts[2] == "output":
                            self.output_layer_indices.append(layer_index)
                        elif parts[2] == "input":
                            self.input_layer_indices.append(layer_index)
                        else:
                            raise ValueError(
                                f"Invalid layer feature name: {feature_name}"
                            )
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid layer index in feature name: {feature_name}"
                        ) from e

        combined = sorted(
            set(self.input_layer_indices) | set(self.output_layer_indices)
        )
        self.layer_indices = combined

    def get_features(self, num_layers: int) -> list[LayerHookResult | None]:
        input_features: list[Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | None] = [
            None
        ] * num_layers
        output_features: list[Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | None] = [
            None
        ] * num_layers

        for layer_index, hook in zip(self.layer_indices, self.layer_hooks):
            assert num_layers > layer_index, (
                f"Layer index {layer_index} out of range for model with {num_layers} layers"
            )
            if hook.result is None:
                continue
            if layer_index in self.input_layer_indices:
                input_features[layer_index] = hook.result.hidden_states
            if layer_index in self.output_layer_indices:
                output_features[layer_index] = hook.result.hidden_states_output

        features: list[LayerHookResult | None] = []
        for layer_input, layer_output in zip(input_features, output_features):
            if layer_input is None and layer_output is None:
                features.append(None)
            else:
                features.append(
                    LayerHookResult(input=layer_input, output=layer_output)
                )

        return features
