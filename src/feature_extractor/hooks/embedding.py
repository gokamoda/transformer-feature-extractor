from dataclasses import dataclass

from transformers import PreTrainedModel

from feature_extractor.configs import FeatureConfig
from feature_extractor.configs.schema import EmbeddingFeatureSpec
from feature_extractor.hooks.base import Hook
from feature_extractor.models import BaseModelArchitecture
from feature_extractor.typing import BATCH, HIDDEN_DIM, SEQUENCE, Tensor

from .base import AbstractBatchResult


@dataclass(repr=False, init=False)
class BatchEmbeddingObservationResult(AbstractBatchResult):
    embeddings: Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class EmbeddingHook(Hook):
    result: BatchEmbeddingObservationResult

    def save_result(self, hook_result: dict):
        self.result = BatchEmbeddingObservationResult(**hook_result)


@dataclass
class EmbeddingHookResult:
    output: None | Tensor[BATCH, SEQUENCE, HIDDEN_DIM]


class EmbeddingHookManager:
    embedding_hook: EmbeddingHook | None

    def __init__(
        self,
        model: PreTrainedModel,
        architecture: BaseModelArchitecture,
        feature_cfg: FeatureConfig,
    ):
        self.model_architecture = architecture
        self.feature_cfg = feature_cfg
        self.embedding_hook = None
        self.install_hook(model)

    def install_hook(self, model: PreTrainedModel):
        model_module = getattr(model, self.model_architecture.model_field)
        embedding_module = getattr(
            model_module,
            self.model_architecture.word_embedding_field,
        )
        self.embedding_hook = EmbeddingHook(
            module=embedding_module,
            to_cpu=True,
            with_output=["embeddings"],
        )

    def remove_hooks(self):
        if self.embedding_hook is not None:
            self.embedding_hook.remove()
            self.embedding_hook = None

    @staticmethod
    def need_embedding_hook(feature_cfg: FeatureConfig) -> bool:
        return any(
            isinstance(feature, EmbeddingFeatureSpec)
            for feature in feature_cfg.feature_specs
        )

    def get_features(self) -> EmbeddingHookResult:
        if self.embedding_hook is None or self.embedding_hook.result is None:
            return EmbeddingHookResult(output=None)
        return EmbeddingHookResult(output=self.embedding_hook.result.embeddings)
