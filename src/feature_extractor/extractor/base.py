from __future__ import annotations

import re
from typing import Any, Generator

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.hooks.results import ExtractorResult, LayerFeatures
from feature_extractor.models.load import load_causal_model, load_tokenizer

_RESIDUAL_FEATURE_RE = re.compile(r"residual\.layer_(\d+)\.(pre_attn|post_ffn)")


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
        self.device = self._resolve_device()
        self.feature_cfg = feature_cfg

    def register_hooks(self):
        # For this basic implementation, we don't need to register any hooks
        pass

    @torch.no_grad()
    def extract_features(
        self,
        data_loader: DataLoader,
    ) -> Generator[ExtractorResult, None, None]:
        feature_plan = self._parse_feature_names()
        expected_num_layers: int | None = None

        self.model.eval()
        for batch in data_loader:
            inputs = self._prepare_batch(batch)
            model_inputs = {
                key: value
                for key, value in inputs.items()
                if self._is_tensor_input(value)
            }
            if "input_ids" not in model_inputs:
                msg = (
                    "Prepared batch does not contain input_ids tensor. "
                    f"Available keys: {sorted(model_inputs.keys())}. "
                    "Ensure the collate function returns input_ids tensors."
                )
                raise ValueError(msg)
            outputs = self.model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                msg = "Model did not return hidden states."
                raise ValueError(msg)

            actual_num_layers = len(hidden_states) - 1
            if expected_num_layers is None:
                expected_num_layers = actual_num_layers
                feature_plan.validate_layer_indices(expected_num_layers)
            elif actual_num_layers != expected_num_layers:
                msg = (
                    "Model returned inconsistent hidden state lengths. "
                    f"Expected {expected_num_layers + 1} hidden states but got "
                    f"{len(hidden_states)}."
                )
                raise ValueError(msg)

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
                yield (
                    ExtractorResult(
                        embeddings=embeddings,
                        layer_features=layer_features,
                        attention_features=[],
                        mlp_features=[],
                    )
                )


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
                msg = (
                    "Unsupported tensor batch length. Expected 1 or 2 tensors, "
                    f"got {len(batch)}."
                )
                raise TypeError(msg)

        msg = f"Unsupported batch type: {type(batch)}"
        raise TypeError(msg)

    def _resolve_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device

        parameter = next(self.model.parameters(), None)
        if parameter is None:
            msg = "Model has no parameters and no device attribute."
            raise ValueError(msg)

        return parameter.device

    def _is_tensor_input(self, value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, (list, tuple)):
            if not value:
                return False
            for item in value:
                if not self._is_tensor_input(item):
                    return False
            return True
        return False

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _build_layer_features(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        sample_index: int,
        feature_plan: _FeaturePlan,
    ) -> list[LayerFeatures]:
        layer_features: list[LayerFeatures] = []
        for layer_idx in feature_plan.sorted_layers:
            input_tensor = (
                hidden_states[layer_idx][sample_index].detach().cpu()
                if layer_idx in feature_plan.pre_attn_layers
                else None
            )
            output_tensor = (
                hidden_states[layer_idx + 1][sample_index].detach().cpu()
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

    def _parse_feature_names(self) -> _FeaturePlan:
        include_embeddings = False
        pre_attn_layers: set[int] = set()
        post_ffn_layers: set[int] = set()
        unknown: list[str] = []

        for feature_name in self.feature_cfg.feature_names:
            if feature_name == "embeddings":
                include_embeddings = True
                continue

            match = _RESIDUAL_FEATURE_RE.fullmatch(feature_name)
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
                f"Requested layer index {max_index} exceeds available layers "
                f"(valid range: 0-{num_layers - 1})."
            )
            raise ValueError(msg)
