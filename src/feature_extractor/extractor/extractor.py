import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TokenizersBackend

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.hooks import HookResult, LayerHookManager, LayerHookResult
from feature_extractor.models import (
    get_model_architecture,
    load_causal_model,
    load_tokenizer,
)


class FeatureExtractor:
    model: PreTrainedModel
    tokenizer: TokenizersBackend
    feature_cfg: FeatureConfig
    layer_hook: LayerHookManager | None = None

    def __init__(
        self,
        model_name_or_path: str,
        feature_cfg: FeatureConfig,
        hook_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = load_causal_model(model_name_or_path)
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.architecture = get_model_architecture(self.model.__class__.__name__)
        self.hook_dtype = hook_dtype
        self.feature_cfg = feature_cfg
        self.install_hooks()

    def install_hooks(self):
        if LayerHookManager.need_layer_hook(self.feature_cfg):
            self.layer_hook = LayerHookManager(
                model=self.model,
                architecture=self.architecture,
                feature_cfg=self.feature_cfg,
            )

    def get_model_num_layers(self):
        return getattr(self.model.config, self.architecture.config_num_layers)

    def get_features(self):
        layer_result: list[LayerHookResult | None] | None = None
        if self.layer_hook is not None:
            layer_result = self.layer_hook.get_features(
                num_layers=self.get_model_num_layers()
            )

        return HookResult(layers=layer_result)

    @torch.no_grad()
    def extract_features(
        self,
        data_loader: DataLoader,
    ):
        for batch in data_loader:
            self.model(
                input_ids=batch["input_ids"].to(self.model.device),
                attention_mask=batch["attention_mask"].to(self.model.device),
            )

            yield batch, self.get_features()
