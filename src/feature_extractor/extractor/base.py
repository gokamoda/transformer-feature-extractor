from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Generator

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.hooks.attention import AttentionHookManager
from feature_extractor.hooks.results import (
    AttentionFeatures,
    ExtractorResult,
    LayerFeatures,
    MLPFeatures,
)
from feature_extractor.models.load import load_causal_model, load_tokenizer

_RESIDUAL_FEATURE_RE = re.compile(r"residual\.layer_(\d+)\.(pre_attn|post_ffn)")
_LAYER_FEATURE_RE = re.compile(r"layer\.layer_(\d+)\.(attn_output|ffn_output|output)")
_ATTN_FEATURE_RE = re.compile(r"attn\.layer_(\d+)\.(query|key|value|qk_logits|weights)")
_MLP_FEATURE_RE = re.compile(r"mlp\.layer_(\d+)\.activation")
_MAX_TENSOR_NESTING_DEPTH = 3
_logger = logging.getLogger(__name__)


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
        attention_hooks: AttentionHookManager | None = None
        if feature_plan.needs_qkv:
            attention_hooks = AttentionHookManager(self.model)
            attention_hooks.install()

        self.model.eval()
        try:
            for batch in data_loader:
                if attention_hooks is not None:
                    attention_hooks.reset()
                inputs = self._prepare_batch(batch)
                input_keys = sorted(inputs.keys())
                model_inputs = {
                    key: value
                    for key, value in inputs.items()
                    if self._is_tensor_input(value, key=key)
                }
                if "input_ids" not in model_inputs:
                    msg = (
                        "Prepared batch does not contain input_ids tensor. "
                        f"Available tensor keys: {sorted(model_inputs.keys())}. "
                        f"Original keys: {input_keys}. "
                        "Ensure the collate function returns input_ids tensors."
                    )
                    raise ValueError(msg)
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    output_attentions=feature_plan.needs_attentions,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    msg = "Model did not return hidden states."
                    raise ValueError(msg)
                attentions = (
                    outputs.attentions if feature_plan.needs_attentions else None
                )

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
                missing_attentions = attentions is None
                is_sequence = isinstance(attentions, Sequence) and not isinstance(
                    attentions, (str, bytes)
                )
                if is_sequence:
                    if len(attentions) == 0:
                        missing_attentions = True
                        attentions = None
                elif attentions is not None:
                    _logger.warning(
                        "Model returned attention weights in unsupported format "
                        f"{type(attentions)}; returning None for attention weight "
                        "features."
                    )
                    missing_attentions = True
                    attentions = None
                if feature_plan.needs_attentions and missing_attentions:
                    _logger.warning(
                        "Model did not return attention weights; returning None "
                        "for attention weight features."
                    )
                if is_sequence and len(attentions) != actual_num_layers:
                    msg = (
                        "Model returned inconsistent attention lengths. "
                        f"Expected {actual_num_layers} attention tensors but got "
                        f"{len(attentions)}."
                    )
                    raise ValueError(msg)
                if attention_hooks is not None:
                    attention_hooks.validate_layer_count(actual_num_layers)

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
                    attention_features = self._build_attention_features(
                        attentions,
                        attention_hooks,
                        idx,
                        feature_plan,
                    )
                    mlp_features = self._build_mlp_features(feature_plan)
                    yield (
                        ExtractorResult(
                            embeddings=embeddings,
                            layer_features=layer_features,
                            attention_features=attention_features,
                            mlp_features=mlp_features,
                        )
                    )
        finally:
            if attention_hooks is not None:
                attention_hooks.remove()


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

    def _is_tensor_input(
        self, value: Any, depth: int = 0, key: str | None = None
    ) -> bool:
        """Return True for tensor-like inputs passed to the model.

        Parameters
        ----------
        value : Any
            Candidate input value to inspect.
        depth : int
            Current nesting depth for list/tuple inspection.
        key : str | None
            Batch key name for logging context when filtering nested structures.

        Returns
        -------
        bool
            True when the value is a tensor or a nested list/tuple of tensors
            within the allowed nesting depth.
        """
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, (list, tuple)):
            if not value:
                # Empty sequences provide no tensor payload to forward.
                return False
            if depth >= _MAX_TENSOR_NESTING_DEPTH:
                _logger.warning(
                    "Skipping nested tensor input for key '%s' deeper than %d levels.",
                    key,
                    _MAX_TENSOR_NESTING_DEPTH,
                )
                return False
            return all(
                self._is_tensor_input(item, depth + 1, key=key) for item in value
            )
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
            layer_output = (
                hidden_states[layer_idx + 1][sample_index].detach().cpu()
                if layer_idx in feature_plan.output_or_ffn_layers
                else None
            )
            output_tensor = (
                layer_output if layer_idx in feature_plan.output_layers else None
            )
            input_tensor = (
                hidden_states[layer_idx][sample_index].detach().cpu()
                if layer_idx in feature_plan.pre_attn_layers
                else None
            )
            mlp_output = (
                layer_output if layer_idx in feature_plan.ffn_output_layers else None
            )
            layer_features.append(
                LayerFeatures(
                    input=input_tensor,
                    attn_output=None,
                    mlp_output=mlp_output,
                    output=output_tensor,
                )
            )
        return layer_features

    def _build_attention_features(
        self,
        attentions: tuple[torch.Tensor, ...] | None,
        attention_hooks: AttentionHookManager | None,
        sample_index: int,
        feature_plan: _FeaturePlan,
    ) -> list[AttentionFeatures]:
        attention_features: list[AttentionFeatures] = []
        for layer_idx in feature_plan.sorted_attention_layers:
            query = None
            key = None
            value = None
            qk_logits = None
            if feature_plan.needs_qkv:
                if attention_hooks is None:
                    msg = (
                        "Attention query/key/value features require projection hooks."
                    )
                    raise ValueError(msg)
                if (
                    layer_idx in feature_plan.attn_query_layers
                    or layer_idx in feature_plan.attn_qk_logits_layers
                ):
                    query = attention_hooks.query(layer_idx, sample_index)
                if (
                    layer_idx in feature_plan.attn_key_layers
                    or layer_idx in feature_plan.attn_qk_logits_layers
                ):
                    key = attention_hooks.key(layer_idx, sample_index)
                if layer_idx in feature_plan.attn_value_layers:
                    value = attention_hooks.value(layer_idx, sample_index)
                if layer_idx in feature_plan.attn_qk_logits_layers:
                    if query is None or key is None:
                        msg = (
                            "Attention qk_logits requested but projections were missing."
                        )
                        raise ValueError(msg)
                    qk_logits = attention_hooks.qk_logits(query, key)
            weights = (
                attentions[layer_idx][sample_index].detach().cpu()
                if attentions is not None
                and layer_idx in feature_plan.attn_weights_layers
                else None
            )
            attention_features.append(
                AttentionFeatures(
                    query=query,
                    key=key,
                    value=value,
                    qk_logits=qk_logits,
                    attn_weights=weights,
                )
            )
        return attention_features

    def _build_mlp_features(self, feature_plan: _FeaturePlan) -> list[MLPFeatures]:
        return [MLPFeatures(activation=None) for _ in feature_plan.sorted_mlp_layers]

    def _parse_feature_names(self) -> _FeaturePlan:
        include_embeddings = False
        pre_attn_layers: set[int] = set()
        post_ffn_layers: set[int] = set()
        layer_attn_output_layers: set[int] = set()
        layer_ffn_output_layers: set[int] = set()
        layer_output_layers: set[int] = set()
        attn_query_layers: set[int] = set()
        attn_key_layers: set[int] = set()
        attn_value_layers: set[int] = set()
        attn_qk_logits_layers: set[int] = set()
        attn_weights_layers: set[int] = set()
        mlp_activation_layers: set[int] = set()
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

            match = _LAYER_FEATURE_RE.fullmatch(feature_name)
            if match:
                layer_index = int(match.group(1))
                feature_kind = match.group(2)
                if feature_kind == "attn_output":
                    layer_attn_output_layers.add(layer_index)
                elif feature_kind == "ffn_output":
                    layer_ffn_output_layers.add(layer_index)
                else:
                    layer_output_layers.add(layer_index)
                continue

            match = _ATTN_FEATURE_RE.fullmatch(feature_name)
            if match:
                layer_index = int(match.group(1))
                feature_kind = match.group(2)
                if feature_kind == "query":
                    attn_query_layers.add(layer_index)
                elif feature_kind == "key":
                    attn_key_layers.add(layer_index)
                elif feature_kind == "value":
                    attn_value_layers.add(layer_index)
                elif feature_kind == "qk_logits":
                    attn_qk_logits_layers.add(layer_index)
                else:
                    attn_weights_layers.add(layer_index)
                continue

            match = _MLP_FEATURE_RE.fullmatch(feature_name)
            if match:
                layer_index = int(match.group(1))
                mlp_activation_layers.add(layer_index)
                continue

            unknown.append(feature_name)

        if unknown:
            msg = f"Unsupported feature names: {', '.join(unknown)}"
            raise ValueError(msg)

        if layer_attn_output_layers:
            _logger.warning(
                "Attention output features are not captured in the minimal "
                "extractor and will be returned as None."
            )
        if mlp_activation_layers:
            _logger.warning(
                "MLP activation features are not captured in the minimal "
                "extractor and will be returned as None."
            )

        return _FeaturePlan(
            include_embeddings=include_embeddings,
            pre_attn_layers=pre_attn_layers,
            post_ffn_layers=post_ffn_layers,
            layer_attn_output_layers=layer_attn_output_layers,
            layer_ffn_output_layers=layer_ffn_output_layers,
            layer_output_layers=layer_output_layers,
            attn_query_layers=attn_query_layers,
            attn_key_layers=attn_key_layers,
            attn_value_layers=attn_value_layers,
            attn_qk_logits_layers=attn_qk_logits_layers,
            attn_weights_layers=attn_weights_layers,
            mlp_activation_layers=mlp_activation_layers,
        )


class _FeaturePlan:
    def __init__(
        self,
        *,
        include_embeddings: bool,
        pre_attn_layers: set[int],
        post_ffn_layers: set[int],
        layer_attn_output_layers: set[int],
        layer_ffn_output_layers: set[int],
        layer_output_layers: set[int],
        attn_query_layers: set[int],
        attn_key_layers: set[int],
        attn_value_layers: set[int],
        attn_qk_logits_layers: set[int],
        attn_weights_layers: set[int],
        mlp_activation_layers: set[int],
    ) -> None:
        self.include_embeddings = include_embeddings
        self.pre_attn_layers = pre_attn_layers
        self.post_ffn_layers = post_ffn_layers
        self.layer_attn_output_layers = layer_attn_output_layers
        self.layer_ffn_output_layers = layer_ffn_output_layers
        self.layer_output_layers = layer_output_layers
        self.attn_query_layers = attn_query_layers
        self.attn_key_layers = attn_key_layers
        self.attn_value_layers = attn_value_layers
        self.attn_qk_logits_layers = attn_qk_logits_layers
        self.attn_weights_layers = attn_weights_layers
        self.mlp_activation_layers = mlp_activation_layers
        self.sorted_layers = sorted(
            pre_attn_layers
            | post_ffn_layers
            | layer_attn_output_layers
            | layer_ffn_output_layers
            | layer_output_layers
        )
        self.sorted_attention_layers = sorted(
            attn_query_layers
            | attn_key_layers
            | attn_value_layers
            | attn_qk_logits_layers
            | attn_weights_layers
        )
        self.sorted_mlp_layers = sorted(mlp_activation_layers)
        self.all_layers = (
            set(self.sorted_layers)
            | set(self.sorted_attention_layers)
            | set(self.sorted_mlp_layers)
        )
        self.output_layers = post_ffn_layers | layer_output_layers
        self.ffn_output_layers = layer_ffn_output_layers
        self.output_or_ffn_layers = self.output_layers | self.ffn_output_layers

    def validate_layer_indices(self, num_layers: int) -> None:
        if not self.all_layers:
            return
        max_index = max(self.all_layers)
        if max_index >= num_layers:
            msg = (
                f"Requested layer index {max_index} exceeds available layers "
                f"(valid range: 0-{num_layers - 1})."
            )
            raise ValueError(msg)

    @property
    def needs_attentions(self) -> bool:
        return bool(self.attn_weights_layers)

    @property
    def needs_qkv(self) -> bool:
        return bool(
            self.attn_query_layers
            or self.attn_key_layers
            or self.attn_value_layers
            or self.attn_qk_logits_layers
        )
