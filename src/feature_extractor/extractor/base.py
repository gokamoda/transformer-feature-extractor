from __future__ import annotations

import re
from typing import Any

import torch
from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.hooks.results import ExtractorResult, LayerFeatures
from feature_extractor.models.load import load_causal_model, load_tokenizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseFeatureExtractor:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: str

    def __init__(
        self,
        model_name_or_path: str,
        feature_cfg: FeatureConfig,
    ) -> None:
        self.model = load_causal_model(model_name_or_path)
        self.tokenizer = load_tokenizer(model_name_or_path)
        self.device = self.model.device
        self.feature_cfg = feature_cfg


    def extract_features(
        self,
        data_loader: DataLoader,
    ) -> list[ExtractorResult]:
        feature_plan = self._parse_feature_names()
        results: list[ExtractorResult] = []

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = self._prepare_batch(batch)
                outputs = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    msg = "Model did not return hidden states."
                    raise ValueError(msg)

                num_layers = len(hidden_states) - 1
                feature_plan.validate_layer_indices(num_layers)

                batch_size = hidden_states[0].shape[0]
                for idx in range(batch_size):
                    embeddings = (
                        hidden_states[0][idx].detach().cpu()
                        if feature_plan.include_embeddings
                        else None
                    )
                    layer_features = self._build_layer_features(
                        hidden_states, idx, feature_plan
                    )
                    results.append(
                        ExtractorResult(
                            embeddings=embeddings,
                            layer_features=layer_features,
                            attention_features=[],
                            mlp_features=[],
                        )
                    )

        return results

    def _prepare_batch(self, batch: Any) -> dict[str, Any]:
        if isinstance(batch, dict):
            return self._move_to_device(batch)

        if isinstance(batch, torch.Tensor):
            return {"input_ids": batch.to(self.device)}

        if isinstance(batch, (list, tuple)):
            if batch and isinstance(batch[0], str):
                encoded = self.tokenizer(
                    list(batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                return self._move_to_device(dict(encoded))

            if batch and all(isinstance(item, torch.Tensor) for item in batch):
                if len(batch) == 1:
                    return {"input_ids": batch[0].to(self.device)}
                if len(batch) == 2:
                    return {
                        "input_ids": batch[0].to(self.device),
                        "attention_mask": batch[1].to(self.device),
                    }

        msg = f"Unsupported batch type: {type(batch)}"
        raise TypeError(msg)

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _build_layer_features(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        batch_index: int,
        feature_plan: _FeaturePlan,
    ) -> list[LayerFeatures]:
        layer_features: list[LayerFeatures] = []
        for layer_idx in feature_plan.sorted_layers:
            input_tensor = (
                hidden_states[layer_idx][batch_index].detach().cpu()
                if layer_idx in feature_plan.pre_attn_layers
                else None
            )
            output_tensor = (
                hidden_states[layer_idx + 1][batch_index].detach().cpu()
                if layer_idx in feature_plan.post_ffn_layers
                else None
            )
            layer_features.append(
                LayerFeatures(
                    input=input_tensor,
                    attn_output=None,
                    mlp_output=None,
                    output=output_tensor,
                )
            )
        return layer_features

    def _parse_feature_names(self) -> "_FeaturePlan":
        include_embeddings = False
        pre_attn_layers: set[int] = set()
        post_ffn_layers: set[int] = set()
        unknown: list[str] = []

        for feature_name in self.feature_cfg.feature_names:
            if feature_name == "embeddings":
                include_embeddings = True
                continue

            match = re.fullmatch(
                r"residual\.layer_(\d+)\.(pre_attn|post_ffn)", feature_name
            )
            if match:
                layer_index = int(match.group(1))
                if match.group(2) == "pre_attn":
                    pre_attn_layers.add(layer_index)
                else:
                    post_ffn_layers.add(layer_index)
                continue

            unknown.append(feature_name)

        if unknown:
            msg = f"Unsupported feature names: {', '.join(unknown)}"
            raise ValueError(msg)

        return _FeaturePlan(
            include_embeddings=include_embeddings,
            pre_attn_layers=pre_attn_layers,
            post_ffn_layers=post_ffn_layers,
        )


class _FeaturePlan:
    def __init__(
        self,
        *,
        include_embeddings: bool,
        pre_attn_layers: set[int],
        post_ffn_layers: set[int],
    ) -> None:
        self.include_embeddings = include_embeddings
        self.pre_attn_layers = pre_attn_layers
        self.post_ffn_layers = post_ffn_layers
        self.sorted_layers = sorted(pre_attn_layers | post_ffn_layers)

    def validate_layer_indices(self, num_layers: int) -> None:
        if not self.sorted_layers:
            return
        max_index = max(self.sorted_layers)
        if max_index >= num_layers:
            msg = (
                f"Requested layer {max_index} but model has {num_layers} layers."
            )
            raise ValueError(msg)
        
