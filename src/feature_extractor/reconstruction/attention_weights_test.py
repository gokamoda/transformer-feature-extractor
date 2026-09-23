import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS, get_attn_scale
from feature_extractor.models.architecture import BaseModelArchitecture
from feature_extractor.models.gemma3 import Gemma3Architecture
from feature_extractor.models.get_config import get_hidden_size_per_head
from feature_extractor.models.get_modules import get_k_norm_module, get_q_norm_module
from feature_extractor.reconstruction.attention_weights import (
    reconstruct_attention_weights,
)


# Above this many parameters, casting to fp32 would double memory usage
# enough to risk OOM (e.g. Llama-2-7b-hf needs ~28GB in fp32 for weights
# alone). Those large checkpoints weren't failing in bf16 to begin with, so
# skip the cast rather than force it universally.
_MAX_PARAMS_FOR_FP32 = 2_000_000_000


def _maybe_float(model: torch.nn.Module) -> torch.nn.Module:
    """Cast to fp32 for numerical-correctness comparisons, unless the model
    is too large to afford it."""
    num_params = sum(p.numel() for p in model.parameters())
    if num_params <= _MAX_PARAMS_FOR_FP32:
        return model.float()
    return model


def _create_feature_config():
    return FeatureConfig.from_str(
        feature_names=[
            "layers.layer_00.output",
            "attn.layer_01.query",
            "attn.layer_01.key",
            "attn.layer_01.value",
            "attn.layer_01.attn_weights",
            "attn.layer_01.attention_mask",
            "attn.layer_01.positional_embedding",
            "attn.layer_01.output",
        ],
        batch_size=16,
    )


def _create_dataset():
    return TextDataset(
        data=[
            TextDataEntry(idx="0", text="Hello, world!"),
            TextDataEntry(idx="1", text="Testing attention reconstruction."),
        ]
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_attention_weight_reconstruction_accuracy(model_name):
    config = _create_feature_config()
    extractor = FeatureExtractor(model_name_or_path=model_name)
    # Most SUPPORTED_MODELS checkpoints load as bf16 by default; use fp32 so
    # this checks the reconstruction formula, not bf16 rounding noise (see
    # attention_dissection_test.py's ov_combined test for the full rationale).
    extractor.model = _maybe_float(extractor.model)
    extractor.configure(config)

    dataset = _create_dataset()
    collator = create_collator(extractor.tokenizer)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collator,
    )

    batch, hook_result = next(extractor.extract_features(dataloader))
    assert "input_ids" in batch
    attn_result = hook_result.attn[1]
    assert attn_result is not None
    assert attn_result.query is not None
    assert attn_result.key is not None
    assert attn_result.attn_weights is not None
    if extractor.architecture.attn_position_embeddings_arg_name is not None:
        assert attn_result.position_embeddings is not None
    else:
        assert attn_result.position_embeddings is None

    query = attn_result.query
    key = attn_result.key

    # attn_result.query/key are captured on the raw q_proj/k_proj output,
    # before any per-head RMSNorm the architecture may apply (e.g. Qwen3/
    # Gemma3's q_norm/k_norm). Apply it here so it matches what the model fed
    # into RoPE/attention.
    if extractor.architecture.attn_q_norm_field is not None:
        q_norm_module = get_q_norm_module(
            architecture=extractor.architecture,
            layer_index=1,
            model=extractor.model,
        ).eval()
        with torch.no_grad():
            query = q_norm_module(query.to(q_norm_module.weight.device))
    if extractor.architecture.attn_k_norm_field is not None:
        k_norm_module = get_k_norm_module(
            architecture=extractor.architecture,
            layer_index=1,
            model=extractor.model,
        ).eval()
        with torch.no_grad():
            key = k_norm_module(key.to(k_norm_module.weight.device))

    scale = get_attn_scale(
        model_config=extractor.model.config,
        architecture=extractor.architecture,
        head_dim=get_hidden_size_per_head(
            model_config=extractor.model.config, architecture=extractor.architecture
        ),
    )
    reconstructed = reconstruct_attention_weights(
        query=query,
        key=key,
        attention_mask=attn_result.attention_mask,
        position_embeddings=attn_result.position_embeddings,
        attn_use_rope=extractor.architecture.attn_use_rope,
        scale=scale,
    )

    torch.testing.assert_close(
        reconstructed, attn_result.attn_weights, rtol=1e-4, atol=1e-4
    )


def test_attention_reconstruction_requires_rope_embeddings():
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 2, 4)
    with pytest.raises(
        ValueError,
        match="RoPE-based architectures require position embeddings.",
    ):
        reconstruct_attention_weights(
            query=query,
            key=key,
            attention_mask=None,
            position_embeddings=None,
            attn_use_rope=True,
        )


class _MockGemma3Config:
    """A minimal stand-in config where query_pre_attn_scalar deliberately
    differs from head_dim -- unlike gemma-3-1b-pt, where the two happen to
    coincide (both 256) and would silently mask a head_dim-based scale bug.
    """

    query_pre_attn_scalar = 128
    head_dim = 256


def test_get_attn_scale_uses_query_pre_attn_scalar_for_gemma3():
    architecture = Gemma3Architecture()
    model_config = _MockGemma3Config()

    scale = get_attn_scale(
        model_config=model_config, architecture=architecture, head_dim=256
    )

    assert scale == pytest.approx(1 / 128**0.5)
    assert scale != pytest.approx(1 / 256**0.5)


def test_get_attn_scale_defaults_to_head_dim_without_scaling_field():
    architecture = BaseModelArchitecture()
    assert architecture.attn_scaling_field is None

    scale = get_attn_scale(
        model_config=_MockGemma3Config(), architecture=architecture, head_dim=64
    )

    assert scale == pytest.approx(1 / 64**0.5)


def test_reconstruct_attention_weights_uses_query_pre_attn_scalar_scale():
    """Wiring check: reconstruct_attention_weights must actually use the
    Gemma3 query_pre_attn_scalar-derived scale, not silently fall back to
    1/sqrt(head_dim), when both are passed the *same* query/key.
    """
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 1, 2, 4, 256
    query = torch.randn(batch, heads, seq_len, head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim)

    architecture = Gemma3Architecture()
    model_config = _MockGemma3Config()
    gemma3_scale = get_attn_scale(
        model_config=model_config, architecture=architecture, head_dim=head_dim
    )

    reconstructed_with_gemma3_scale = reconstruct_attention_weights(
        query=query,
        key=key,
        attention_mask=None,
        position_embeddings=None,
        attn_use_rope=False,
        scale=gemma3_scale,
    )
    reconstructed_with_head_dim_scale = reconstruct_attention_weights(
        query=query,
        key=key,
        attention_mask=None,
        position_embeddings=None,
        attn_use_rope=False,
    )

    assert not torch.allclose(
        reconstructed_with_gemma3_scale, reconstructed_with_head_dim_scale
    )

    expected_scores = torch.matmul(query, key.transpose(-1, -2)) * gemma3_scale
    expected = torch.softmax(expected_scores, dim=-1)
    torch.testing.assert_close(reconstructed_with_gemma3_scale, expected)
