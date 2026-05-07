from dataclasses import dataclass

from torch import nn
from transformers import PreTrainedModel

from feature_extractor.configs import FeatureConfig
from feature_extractor.configs.schema import LayerFeatureSpec
from feature_extractor.hooks.base import Hook
from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HIDDEN_DIM, SEQUENCE, Tensor

from .base import AbstractBatchResult, StopForwardError


@dataclass(repr=False, init=False)
class BatchLayerObservationResult(AbstractBatchResult):
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]
    hidden_states_output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class LayerHook(Hook):
    result: BatchLayerObservationResult
    early_stop: bool = False

    def __init__(
        self,
        module: nn.Module,
        to_cpu: bool = True,
        with_args: None | list[str] = None,
        with_kwargs: bool | list[str] = False,
        with_output: None | list[str] = None,
        early_stop: bool = False,
        empty_hook: bool = False,
    ):
        super().__init__(
            module=module,
            to_cpu=to_cpu,
            with_args=with_args,
            with_kwargs=with_kwargs,
            with_output=with_output,
        )
        self.early_stop = early_stop
        self.empty_hook = empty_hook

    def save_result(self, hook_result: dict):
        if not self.empty_hook:
            self.result = BatchLayerObservationResult(**hook_result)
        if self.early_stop:
            print("Early stopping triggered by LayerHook.")
            raise StopForwardError(
                "Early stopping forward pass after collecting required layer features."
            )


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
        deepest_layer_index: int | None = None,
    ):
        self.model_architecture = architecture
        self.feature_cfg = feature_cfg
        self.deepest_layer_index = deepest_layer_index
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

        early_exit_applied = False

        for index in self.layer_indices:
            hook_kwargs = {}
            if index in self.input_layer_indices:
                hook_kwargs["with_args"] = self.model_architecture.layers_pos_args
                hook_kwargs["with_kwargs"] = [
                    self.model_architecture.layers_input_hidden_state_arg_name
                ]
            if index in self.output_layer_indices:
                hook_kwargs["with_output"] = self.model_architecture.layer_return_fields

            if index == self.deepest_layer_index:
                hook_kwargs["early_stop"] = True
                assert max(self.layer_indices) == index, (
                    "Early stopping can only be applied to the deepest layer."
                )
                print(f"Applying early stopping at layer index {index}.")
                early_exit_applied = True

            self.layer_hooks.append(
                LayerHook(module=layers_module[index], to_cpu=True, **hook_kwargs)
            )
        if self.deepest_layer_index is not None and not early_exit_applied:
            print(f"Applying early stopping at layer index {self.deepest_layer_index}.")
            self.layer_hooks.append(
                LayerHook(
                    module=layers_module[self.deepest_layer_index],
                    to_cpu=True,
                    early_stop=True,
                    empty_hook=True,
                )
            )

    def remove_hooks(self):
        for hook in self.layer_hooks:
            hook.remove()

    @staticmethod
    def need_layer_hook(feature_cfg: FeatureConfig) -> bool:
        return any(
            isinstance(feature, LayerFeatureSpec)
            for feature in feature_cfg.feature_specs
        )

    def _resolve_layer_index(self, feature_cfg: FeatureConfig) -> None:
        for feature in feature_cfg.feature_specs:
            if not isinstance(feature, LayerFeatureSpec):
                continue
            if feature.feature == "output":
                self.output_layer_indices.append(feature.layer_index)
            elif feature.feature == "input":
                self.input_layer_indices.append(feature.layer_index)

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
                features.append(LayerHookResult(input=layer_input, output=layer_output))

        return features
