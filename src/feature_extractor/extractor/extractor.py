import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TokenizersBackend

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.hooks import (
    AttentionHookManager,
    AttentionHookResult,
    EmbeddingHookManager,
    EmbeddingHookResult,
    HookResult,
    LayerHookManager,
    LayerHookResult,
    MLPHookManager,
    MLPHookResult,
)
from feature_extractor.models import (
    BaseModelArchitecture,
    get_model_architecture,
    get_num_layers,
    load_causal_model,
    load_tokenizer,
)


class FeatureExtractor:
    model: PreTrainedModel
    tokenizer: TokenizersBackend
    architecture: BaseModelArchitecture
    feature_cfg: FeatureConfig
    embedding_hook: EmbeddingHookManager | None = None
    layer_hook: LayerHookManager | None = None
    attn_hook: AttentionHookManager | None = None
    mlp_hook: MLPHookManager | None = None

    def __init__(
        self,
        model_name_or_path: str,
        hook_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = load_causal_model(model_name_or_path)
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.architecture = get_model_architecture(self.model)
        self.hook_dtype = hook_dtype

    def configure(
        self,
        feature_cfg: FeatureConfig,
    ):
        self.feature_cfg = feature_cfg
        self.install_hooks()
        if self.attn_hook is not None and self.attn_hook.need_eager_attn():
            self.model.set_attn_implementation("eager")

    def install_hooks(self):
        if EmbeddingHookManager.need_embedding_hook(self.feature_cfg):
            self.embedding_hook = EmbeddingHookManager(
                model=self.model,
                architecture=self.architecture,
                feature_cfg=self.feature_cfg,
            )

        if LayerHookManager.need_layer_hook(self.feature_cfg):
            if not self.architecture.supports_layer_output:
                raise ValueError(
                    f"Architecture {self.architecture.__class__.__name__} does not support layer output hooks."
                )
            self.layer_hook = LayerHookManager(
                model=self.model,
                architecture=self.architecture,
                feature_cfg=self.feature_cfg,
            )

        if AttentionHookManager.need_attn_hook(self.feature_cfg):
            if not self.architecture.supports_attention_qkv:
                raise ValueError(
                    f"Architecture {self.architecture.__class__.__name__} does not support attention QKV hooks."
                )
            self.attn_hook = AttentionHookManager(
                model=self.model,
                architecture=self.architecture,
                feature_cfg=self.feature_cfg,
            )
        if MLPHookManager.need_mlp_hook(self.feature_cfg):
            if not self.architecture.supports_mlp_output:
                raise ValueError(
                    f"Architecture {self.architecture.__class__.__name__} does not support MLP hooks."
                )
            self.mlp_hook = MLPHookManager(
                model=self.model,
                architecture=self.architecture,
                feature_cfg=self.feature_cfg,
            )

    def get_features(self):
        embedding_result: EmbeddingHookResult | None = None
        layer_result: list[LayerHookResult | None] | None = None
        attn_result: list[AttentionHookResult | None] | None = None
        mlp_result: list[MLPHookResult | None] | None = None
        if self.embedding_hook is not None:
            embedding_result = self.embedding_hook.get_features()
        if self.layer_hook is not None:
            layer_result = self.layer_hook.get_features(
                num_layers=get_num_layers(self.model.config, self.architecture)
            )
        if self.attn_hook is not None:
            attn_result = self.attn_hook.get_features(
                num_layers=get_num_layers(self.model.config, self.architecture)
            )
        if self.mlp_hook is not None:
            mlp_result = self.mlp_hook.get_features(
                num_layers=get_num_layers(self.model.config, self.architecture)
            )
        return HookResult(
            embeddings=embedding_result,
            layers=layer_result,
            attn=attn_result,
            mlp=mlp_result,
        )

    @torch.no_grad()
    def extract_features(
        self,
        data_loader: DataLoader,
    ):
        for batch in data_loader:
            self.model.generate(  # ty: ignore
                input_ids=batch["input_ids"].to(self.model.device),
                attention_mask=batch["attention_mask"].to(self.model.device),
                return_dict_in_generate=True,
                output_attentions=(
                    self.attn_hook is not None and self.attn_hook.need_eager_attn()
                ),
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=1,
            )

            yield batch, self.get_features()
