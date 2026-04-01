from dataclasses import dataclass

from transformers import PreTrainedModel

from feature_extractor.configs import FeatureConfig
from feature_extractor.hooks.base import Hook
from feature_extractor.models import BaseModelArchitecture
from feature_extractor.models.architecture import get_num_layers
from feature_extractor.typing import BATCH, HIDDEN_DIM, MLP_DIM, SEQUENCE, Tensor

from .base import AbstractBatchResult


@dataclass(repr=False, init=False)
class MLPActivationObservationResult(AbstractBatchResult):
    activation: Tensor[BATCH, SEQUENCE, MLP_DIM]


class MLPActivationHook(Hook):
    result: MLPActivationObservationResult

    def save_result(self, hook_result: dict):
        self.result = MLPActivationObservationResult(**hook_result)


@dataclass(repr=False, init=False)
class MLPObservationResult(AbstractBatchResult):
    output: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class MLPHook(Hook):
    result: MLPObservationResult

    def save_result(self, hook_result: dict):
        self.result = MLPObservationResult(**hook_result)


@dataclass
class MLPHookResult:
    activation: None | Tensor[BATCH, SEQUENCE, MLP_DIM]
    down_proj_input: None | Tensor[BATCH, SEQUENCE, MLP_DIM]
    output: None | Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class MLPHookManager:
    activation_layer_indices: list[int]
    down_proj_input_layer_indices: list[int]
    output_layer_indices: list[int]
    activation_down_proj_input_output_combined_layer_indices: list[int]
    activation_hooks: list[MLPActivationHook]
    down_proj_input_hooks: list[MLPActivationHook]
    output_hooks: list[MLPHook]

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
        self.check_layer_index_in_range(model)
        self.install_hooks(model)

    def reset_hooks(self):
        self.activation_layer_indices = []
        self.down_proj_input_layer_indices = []
        self.output_layer_indices = []
        self.activation_down_proj_input_output_combined_layer_indices = []
        self.activation_hooks = []
        self.down_proj_input_hooks = []
        self.output_hooks = []

    @staticmethod
    def need_mlp_hook(feature_cfg: FeatureConfig) -> bool:
        for feature_name in feature_cfg.feature_names:
            if feature_name.startswith("mlp."):
                return True
        return False

    def _resolve_layer_index(self, feature_cfg: FeatureConfig):
        for feature_name in feature_cfg.feature_names:
            if feature_name.startswith("mlp."):
                parts = feature_name.split(".")
                if len(parts) == 3 and parts[1].startswith("layer_"):
                    try:
                        layer_index = int(parts[1].split("_")[1])
                        if parts[2] == "activation":
                            self.activation_layer_indices.append(layer_index)
                        elif parts[2] == "down_proj_input":
                            self.down_proj_input_layer_indices.append(layer_index)
                        elif parts[2] == "output":
                            self.output_layer_indices.append(layer_index)
                        else:
                            raise ValueError(
                                f"Invalid MLP feature name: {feature_name}"
                            )
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid layer index in feature name: {feature_name}"
                        ) from e

        if (
            len(self.activation_layer_indices) > 0
            or len(self.down_proj_input_layer_indices) > 0
            or len(self.output_layer_indices) > 0
        ):
            layer_indices = list(
                set(self.activation_layer_indices)
                | set(self.down_proj_input_layer_indices)
                | set(self.output_layer_indices)
            )
            layer_indices.sort()
            self.activation_down_proj_input_output_combined_layer_indices = layer_indices

    def check_layer_index_in_range(self, model: PreTrainedModel):
        num_layers = get_num_layers(model.config, self.model_architecture)
        for layer_index in self.activation_down_proj_input_output_combined_layer_indices:
            if layer_index < 0 or layer_index >= num_layers:
                raise ValueError(
                    f"Layer index {layer_index} is out of range for model with {num_layers} layers."
                )

    def install_hooks(self, model: PreTrainedModel):
        layers_module = getattr(
            getattr(model, self.model_architecture.model_field),
            self.model_architecture.layers_field,
        )
        for layer_index in self.activation_layer_indices:
            mlp_module = getattr(
                layers_module[layer_index], self.model_architecture.mlp_field
            )
            self.activation_hooks.append(
                MLPActivationHook(
                    module=getattr(
                        mlp_module, self.model_architecture.mlp_activation_field
                    ),
                    to_cpu=True,
                    with_output=["activation"],
                )
            )

        for layer_index in self.output_layer_indices:
            mlp_module = getattr(
                layers_module[layer_index], self.model_architecture.mlp_field
            )
            self.output_hooks.append(
                MLPHook(
                    module=mlp_module,
                    to_cpu=True,
                    with_output=["output"],
                )
            )
        for layer_index in self.down_proj_input_layer_indices:
            mlp_module = getattr(
                layers_module[layer_index], self.model_architecture.mlp_field
            )
            self.down_proj_input_hooks.append(
                MLPActivationHook(
                    module=getattr(mlp_module, self.model_architecture.mlp_down_proj_field),
                    to_cpu=True,
                    with_args=["activation"],
                )
            )

    def remove_hooks(self):
        for hook in self.activation_hooks:
            hook.remove()
        for hook in self.down_proj_input_hooks:
            hook.remove()
        for hook in self.output_hooks:
            hook.remove()

    def get_features(self, num_layers: int) -> list[MLPHookResult | None]:
        features = []
        activation_features: list[Tensor[BATCH, SEQUENCE, MLP_DIM] | None] = [
            None
        ] * num_layers
        down_proj_input_features: list[Tensor[BATCH, SEQUENCE, MLP_DIM] | None] = [
            None
        ] * num_layers
        output_features: list[Tensor[BATCH, SEQUENCE, HIDDEN_DIM] | None] = [
            None
        ] * num_layers

        for layer_index, hook in zip(
            self.activation_layer_indices, self.activation_hooks
        ):
            activation_features[layer_index] = hook.result.activation
        for layer_index, hook in zip(
            self.down_proj_input_layer_indices, self.down_proj_input_hooks
        ):
            down_proj_input_features[layer_index] = hook.result.activation
        for layer_index, hook in zip(self.output_layer_indices, self.output_hooks):
            output_features[layer_index] = hook.result.output

        for activation, down_proj_input, output in zip(
            activation_features, down_proj_input_features, output_features
        ):
            if activation is None and down_proj_input is None and output is None:
                features.append(None)
            else:
                features.append(
                    MLPHookResult(
                        activation=activation,
                        down_proj_input=down_proj_input,
                        output=output,
                    )
                )

        return features
