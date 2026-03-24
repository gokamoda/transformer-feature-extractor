from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.extractor.base import BaseFeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS


RUN_SUPPORTED_MODELS = os.getenv("RUN_SUPPORTED_MODELS") == "1"


def _has_hf_token() -> bool:
    return bool(os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN"))


def _requires_hf_token(model_name: str) -> bool:
    return model_name.startswith("meta-llama")


def _requires_gpu(model_name: str) -> bool:
    return model_name == "meta-llama/Llama-2-7b-hf"


@pytest.mark.skipif(
    not RUN_SUPPORTED_MODELS,
    reason="Set RUN_SUPPORTED_MODELS=1 to run integration tests.",
)
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_supported_models_extract_embeddings(model_name: str) -> None:
    if _requires_hf_token(model_name) and not _has_hf_token():
        pytest.skip("HF token required for meta-llama models.")
    if _requires_gpu(model_name) and not torch.cuda.is_available():
        pytest.skip("Llama-2-7b-hf requires GPU to run in CI.")

    feature_cfg = FeatureConfig(feature_names=["embeddings"])
    extractor = BaseFeatureExtractor(model_name, feature_cfg)
    data_loader = DataLoader(["Hello world"], batch_size=1)

    results = list(extractor.extract_features(data_loader))

    assert len(results) == 1
    assert results[0].embeddings is not None
