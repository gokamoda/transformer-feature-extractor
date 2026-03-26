from dataclasses import dataclass

from feature_extractor.typing import BATCH, HIDDEN_DIM, SEQUENCE, Tensor
from feature_extractor.configs import FeatureConfig
from feature_extractor.models import BaseModelArchitecture
from transformers import PreTrainedModel
from feature_extractor.hooks.base import Hook

from .base import AbstractBatchResult


@dataclass(repr=False, init=False)
class BatchHiddenStateObservationResult(AbstractBatchResult):
    hidden_states: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


@dataclass
class LayerHookResult:
    output: None | Tensor[BATCH, SEQUENCE, HIDDEN_DIM]

class LayerHookManager:
    layer_indices: list[int]
    layer_hooks: list[Hook]


    def __init__(
            self,
            model: PreTrainedModel,
            architecture: BaseModelArchitecture,
            feature_cfg: FeatureConfig
        ):
        self.model_architecture = architecture
        self.feature_cfg = feature_cfg
        self.layer_indices = self._resolve_layer_index(self.feature_cfg)
        self.layer_hooks = []
        self.install_hook(model)

    def install_hook(self, model: PreTrainedModel):
        layers_module = getattr(getattr(model, self.model_architecture.model_field), self.model_architecture.layers_field)
        for index in self.layer_indices:
            self.layer_hooks.append(
                Hook(
                    module=layers_module[index],
                    result_class=BatchHiddenStateObservationResult,
                    to_cpu=True,
                    with_output=self.model_architecture.layer_return_fields
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
    
    @staticmethod
    def _resolve_layer_index(feature_cfg: FeatureConfig) -> list[int]:
        layer_indices = []
        for feature_name in feature_cfg.feature_names:
            if feature_name.startswith("layers."):
                parts = feature_name.split(".")
                if len(parts) == 3 and parts[1].startswith("layer_"):
                    try:
                        layer_index = int(parts[1].split("_")[1])
                        layer_indices.append(int(layer_index))
                    except ValueError:
                        raise ValueError(f"Invalid layer index in feature name: {feature_name}")
        return layer_indices
    
    def get_features(self, num_layers: int) -> list[LayerHookResult | None]:
        features = [None] * num_layers

        for layer_index, hook in zip(self.layer_indices, self.layer_hooks):
            if hook.result is not None:
                features[layer_index] = LayerHookResult(
                    output=hook.result.hidden_states
                )

        return features