import gc
import math

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.hooks import HookResult
from feature_extractor.models import (
    SUPPORTED_MODELS,
    get_model_architecture,
    load_causal_model,
)
from feature_extractor.models.architecture import BaseModelArchitecture
from feature_extractor.models.get_config import get_num_attn_heads, get_num_kv_heads
from feature_extractor.models.get_modules import (
    _get_qkv_proj_module,
    get_k_norm_module,
    get_o_proj_module,
    get_pre_attn_norm_module,
    get_q_norm_module,
    get_rope_module,
    get_v_proj_module,
)
from feature_extractor.reconstruction.attention_dissection import (
    _precompute_qk_weights,
    reconstruct_attn_output_vo_combined,
    reconstruct_attn_weight_qk_combined_norope,
    reconstruct_attn_weight_qk_combined_with_rope,
    reconstruct_qkv_vectors,
    rope_frequency_inner_products,
)
from feature_extractor.reconstruction.attention_weights import _apply_rope
from feature_extractor.reconstruction.rope import SimplifiedRoPEV1
from feature_extractor.typing import HEAD, HIDDEN_DIM, SEQUENCE, Tensor

# Above this many parameters, casting to fp32 would double memory usage
# enough to risk OOM (e.g. Llama-2-7b-hf needs ~28GB in fp32 for weights
# alone). Those large checkpoints weren't failing in bf16 to begin with, so
# skip the cast rather than force it universally.
_MAX_PARAMS_FOR_FP32 = 2_000_000_000


def _maybe_float(model: torch.nn.Module) -> torch.nn.Module:
    """Cast to fp32 for numerical-correctness comparisons (see
    test_attention_reconstruction_accuracy_ov_combined for the rationale),
    unless the model is too large to afford it.
    """
    num_params = sum(p.numel() for p in model.parameters())
    if num_params <= _MAX_PARAMS_FOR_FP32:
        return model.float()
    return model


@pytest.mark.parametrize("scale_by_head_dim", [False, True])
def test_rope_frequency_inner_products_reconstruct_logits(scale_by_head_dim):
    torch.manual_seed(0)
    sequence_length = 5
    head_dim = 8
    query = torch.randn(sequence_length, head_dim, dtype=torch.float64)
    key = torch.randn(sequence_length, head_dim, dtype=torch.float64)
    inv_freq = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float64)
    position_embeddings = SimplifiedRoPEV1(inv_freq).create_position_embeddings(
        sequence_length
    )

    logits_by_frequency = rope_frequency_inner_products(
        query,
        key,
        position_embeddings,
        scale_by_head_dim=scale_by_head_dim,
    )
    query_roped, key_roped = _apply_rope(
        query[None, None], key[None, None], position_embeddings
    )
    expected = query_roped[0, 0] @ key_roped[0, 0].T
    if scale_by_head_dim:
        expected = expected / math.sqrt(head_dim)

    assert logits_by_frequency.shape == (
        head_dim // 2,
        sequence_length,
        sequence_length,
    )
    torch.testing.assert_close(logits_by_frequency.sum(dim=0), expected)


@pytest.mark.parametrize(
    ("query_shape", "key_shape", "error"),
    [
        ((3, 8), (4, 8), "same shape"),
        ((2, 3, 8), (2, 3, 8), "must have shape"),
        ((3, 7), (3, 7), "must be even"),
    ],
)
def test_rope_frequency_inner_products_validates_inputs(
    query_shape, key_shape, error
):
    query = torch.randn(query_shape)
    key = torch.randn(key_shape)
    position_embeddings = (torch.ones(3, 8), torch.zeros(3, 8))

    with pytest.raises(ValueError, match=error):
        rope_frequency_inner_products(query, key, position_embeddings)


def test_rope_frequency_inner_products_zeroed_frequency_is_zero():
    torch.manual_seed(1)
    sequence_length = 4
    head_dim = 8
    frequency_index = 2
    query = torch.randn(sequence_length, head_dim)
    key = torch.randn(sequence_length, head_dim)
    inv_freq = torch.tensor([1.0, 0.1, 0.01, 0.001])
    cos, sin = SimplifiedRoPEV1(inv_freq).create_position_embeddings(
        sequence_length
    )
    cos = cos.clone()
    sin = sin.clone()
    cos[:, [frequency_index, frequency_index + head_dim // 2]] = 0
    sin[:, [frequency_index, frequency_index + head_dim // 2]] = 0

    logits_by_frequency = rope_frequency_inner_products(
        query, key, (cos, sin)
    )

    torch.testing.assert_close(
        logits_by_frequency[frequency_index],
        torch.zeros(sequence_length, sequence_length),
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_rope_frequency_inner_products_reconstruct_model_attention_weights(
    model_name,
):
    extractor = FeatureExtractor(model_name_or_path=model_name)
    if not extractor.architecture.attn_use_rope:
        pytest.skip(f"Model {model_name} does not use RoPE")
    # See test_attention_reconstruction_accuracy_ov_combined: use fp32 so
    # this checks the reconstruction formula, not bf16 rounding noise.
    extractor.model = _maybe_float(extractor.model)
    extractor.configure(
        FeatureConfig.from_str(
            [
                "attn.layer_01.query",
                "attn.layer_01.key",
                "attn.layer_01.attn_weights",
                "attn.layer_01.attention_mask",
                "attn.layer_01.positional_embedding",
            ]
        )
    )

    dataset = _create_dataset()
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=create_collator(extractor.tokenizer),
    )
    _, hook_result = next(extractor.extract_features(dataloader))
    assert hook_result.attn is not None
    attn_result = hook_result.attn[1]
    assert attn_result is not None
    assert attn_result.query is not None
    assert attn_result.key is not None
    assert attn_result.position_embeddings is not None
    assert attn_result.attn_weights is not None

    query = attn_result.query
    key = attn_result.key

    # hook_result.attn[*].query/key are captured on the raw q_proj/k_proj
    # output, before any per-head RMSNorm the architecture may apply (e.g.
    # Qwen3/Gemma3's q_norm/k_norm). Apply it here so the reconstruction
    # below matches what the model actually fed into RoPE/attention.
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

    cos, sin = attn_result.position_embeddings
    logits = []
    for batch_index in range(query.shape[0]):
        batch_logits = []
        if cos.ndim == 3:
            batch_position_embeddings = (cos[batch_index], sin[batch_index])
        else:
            batch_position_embeddings = (cos, sin)
        for head_index in range(query.shape[1]):
            batch_logits.append(
                rope_frequency_inner_products(
                    query[batch_index, head_index],
                    key[batch_index, head_index],
                    (
                        batch_position_embeddings[0],
                        batch_position_embeddings[1],
                    ),
                ).sum(dim=0)
            )
        logits.append(torch.stack(batch_logits))
    reconstructed_logits = torch.stack(logits)

    if attn_result.attention_mask is not None:
        reconstructed_logits = reconstructed_logits + attn_result.attention_mask.to(
            dtype=reconstructed_logits.dtype,
            device=reconstructed_logits.device,
        )
    reconstructed_attn_weights = torch.softmax(reconstructed_logits, dim=-1)

    torch.testing.assert_close(
        reconstructed_attn_weights.cpu(),
        attn_result.attn_weights.cpu(),
        atol=1e-2,
        rtol=1e-2,
    )


def _create_feature_config():
    return FeatureConfig.from_str(
        feature_names=[
            "layers.layer_00.output",
            "attn.layer_01.value",
            "attn.layer_01.query",
            "attn.layer_01.key",
            "attn.layer_01.attn_weights",
            "attn.layer_01.output",
        ],
        batch_size=16,
    )


def _create_dataset():
    return TextDataset(
        data=[
            TextDataEntry(idx="1", text="Testing attention reconstruction."),
            TextDataEntry(idx="0", text="Hello, world!"),
        ]
    )


def _get_result(model_name: str, model: torch.nn.Module | None = None) -> HookResult:
    extractor = FeatureExtractor(model_name_or_path=model_name)
    if model is not None:
        extractor.model = model
    config = _create_feature_config()
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
    return hook_result


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_reconstruct_qkv_vectors(
    model_name,
):
    config = _create_feature_config()
    extractor = FeatureExtractor(model_name_or_path=model_name)
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
    hidden_states = hook_result.layers[0].output

    num_kv_heads = get_num_kv_heads(
        model_config=extractor.model.config, architecture=extractor.architecture
    )
    num_attention_heads = get_num_attn_heads(
        model_config=extractor.model.config, architecture=extractor.architecture
    )

    # layer normalize before attn
    ln_module = get_pre_attn_norm_module(
        architecture=extractor.architecture,
        model=extractor.model,
        layer_index=1,
    ).eval()
    hidden_states = hidden_states.to(ln_module.weight.device)
    with torch.no_grad():
        hidden_states = ln_module(hidden_states)
    del ln_module
    gc.collect()
    torch.cuda.empty_cache()

    # value
    value = hook_result.attn[1].value
    reconstructed_value = (
        reconstruct_qkv_vectors(
            hidden_states=hidden_states,
            qkv_proj_module=_get_qkv_proj_module(
                model=extractor.model,
                architecture=extractor.architecture,
                layer_index=1,
                modules=["v_proj"],
            )["v_proj"].eval(),
            num_kv_heads=num_kv_heads,
            num_attention_heads=num_attention_heads,
            module_type="v_proj",
        )
        .detach()
        .cpu()
    )
    torch.testing.assert_close(reconstructed_value, value, atol=1e-4, rtol=1e-4)

    # query
    query = hook_result.attn[1].query
    reconstructed_query = (
        reconstruct_qkv_vectors(
            hidden_states=hidden_states,
            qkv_proj_module=_get_qkv_proj_module(
                model=extractor.model,
                architecture=extractor.architecture,
                layer_index=1,
                modules=["q_proj"],
            )["q_proj"].eval(),
            num_attention_heads=num_attention_heads,
            module_type="q_proj",
        )
        .detach()
        .cpu()
    )
    torch.testing.assert_close(reconstructed_query, query, atol=1e-4, rtol=1e-4)

    # key
    key = hook_result.attn[1].key
    reconstructed_key = (
        reconstruct_qkv_vectors(
            hidden_states=hidden_states,
            qkv_proj_module=_get_qkv_proj_module(
                model=extractor.model,
                architecture=extractor.architecture,
                layer_index=1,
                modules=["k_proj"],
            )["k_proj"].eval(),
            num_attention_heads=num_attention_heads,
            module_type="k_proj",
            num_kv_heads=num_kv_heads,
        )
        .detach()
        .cpu()
    )
    torch.testing.assert_close(reconstructed_key, key, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_attention_reconstruction_accuracy_ov_combined(model_name):
    architecture = get_model_architecture(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    # Most checkpoints in SUPPORTED_MODELS load as bf16 by default. bf16's
    # ~3 significant digits is fine for a single reconstruction step on most
    # models, but combined with some models' sharper attention (e.g. Gemma3's
    # query_pre_attn_scalar scaling), it can push a handful of elements past
    # a 1e-2 tolerance even though the reconstruction is algebraically
    # correct. Use fp32 so this test checks the formula, not bf16 rounding.
    model = _maybe_float(load_causal_model(model_name))
    hook_result = _get_result(model_name, model=model)
    assert hook_result.attn is not None
    assert hook_result.attn[1] is not None
    assert hook_result.attn[1].attn_weights is not None
    attn_weights = hook_result.attn[1].attn_weights
    assert hook_result.layers is not None
    assert hook_result.layers[0] is not None
    assert hook_result.layers[0].output is not None
    hidden_states = hook_result.layers[0].output
    assert hook_result.attn[1].output is not None
    attn_output = hook_result.attn[1].output

    # layer normalize before attn
    ln_module = get_pre_attn_norm_module(
        architecture=architecture,
        layer_index=1,
        model=model,
    ).eval()
    hidden_states = hidden_states.to(ln_module.weight.device)
    with torch.no_grad():
        hidden_states = ln_module(hidden_states)
    del ln_module
    gc.collect()
    torch.cuda.empty_cache()

    o_proj_module = get_o_proj_module(
        model=model,
        architecture=architecture,
        layer_index=1,
    ).eval()
    v_proj_module = get_v_proj_module(
        model=model,
        architecture=architecture,
        layer_index=1,
    ).eval()

    reconstructed_output = (
        reconstruct_attn_output_vo_combined(
            attn_weights=attn_weights,
            hidden_states=hidden_states,
            v_proj_module=v_proj_module,
            o_proj_module=o_proj_module,
            num_attention_heads=get_num_attn_heads(
                model_config=model_config, architecture=architecture
            ),
            num_kv_heads=get_num_kv_heads(
                model_config=model_config, architecture=architecture
            ),
        )
        .detach()
        .cpu()
    )
    torch.testing.assert_close(reconstructed_output, attn_output, atol=1e-2, rtol=1e-2)


@torch.no_grad()
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_attention_weight_reconstruction_accuracy_qk_combined(model_name):
    model_config = AutoConfig.from_pretrained(model_name)
    architecture = get_model_architecture(model_name)

    if (
        architecture.attn_q_norm_field is not None
        or architecture.attn_k_norm_field is not None
    ):
        pytest.skip(
            f"Model {model_name} applies q_norm/k_norm (RMSNorm) between "
            "q_proj/k_proj and RoPE; _precompute_qk_weights cannot fold that "
            "nonlinear step into a static weight matrix. See "
            "reconstruct_qkv_vectors(norm_module=...) instead."
        )

    # See test_attention_reconstruction_accuracy_ov_combined: use fp32 so
    # this checks the reconstruction formula, not bf16 rounding noise.
    model = _maybe_float(load_causal_model(model_name))
    hook_result = _get_result(model_name, model=model)

    assert hook_result.layers is not None
    assert hook_result.layers[0] is not None
    assert hook_result.layers[0].output is not None
    hidden_states = hook_result.layers[0].output

    num_kv_heads = get_num_kv_heads(
        model_config=model_config, architecture=architecture
    )
    num_attention_heads = get_num_attn_heads(
        model_config=model_config, architecture=architecture
    )

    qk_modules = _get_qkv_proj_module(
        architecture=architecture,
        layer_index=1,
        modules=["q_proj", "k_proj"],
        model=model,
    )
    q_proj_module = qk_modules["q_proj"].eval()
    k_proj_module = qk_modules["k_proj"].eval()
    head_dim = q_proj_module.out_features // num_attention_heads

    # layer normalize before attn
    ln_module = get_pre_attn_norm_module(
        architecture=architecture,
        model=model,
        layer_index=1,
    ).eval()
    hidden_states = hidden_states.to(ln_module.weight.device)
    with torch.no_grad():
        hidden_states = ln_module(hidden_states)
    del ln_module
    gc.collect()
    torch.cuda.empty_cache()

    if architecture.attn_use_rope:
        original_rope_module = get_rope_module(
            model=model,
            architecture=architecture,
        )

        attn_scaling = original_rope_module.attention_scaling
        if isinstance(attn_scaling, torch.Tensor):
            attn_scaling = float(attn_scaling.item())
        elif isinstance(attn_scaling, (int, float)):
            attn_scaling = float(attn_scaling)
        else:
            raise ValueError(
                f"Unexpected type for attention_scaling: {type(attn_scaling)}"
            )

        inv_freq = original_rope_module.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        simplified_rope_module = SimplifiedRoPEV1(
            inv_freq=inv_freq,
            attention_scaling=attn_scaling,
        )

        qk_weight_combined: Tensor[HEAD, SEQUENCE, SEQUENCE, HIDDEN_DIM, HIDDEN_DIM]
        qk_weight_combined, qk_bias_terms = _precompute_qk_weights(
            q_proj_module=q_proj_module,
            k_proj_module=k_proj_module,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            rope_module=simplified_rope_module,
            sequence_length=hidden_states.shape[1],
            architecture=architecture,
        )
        del q_proj_module
        del k_proj_module
        del qk_modules
        del simplified_rope_module
        gc.collect()
        torch.cuda.empty_cache()

        attn_weights = (
            reconstruct_attn_weight_qk_combined_with_rope(
                hidden_states=hidden_states[:1],
                qk_weight_combined=qk_weight_combined,
                qk_bias_terms=qk_bias_terms,
                head_dim=head_dim,
            )
            .detach()
            .cpu()
        )
        assert hook_result.attn is not None
        assert hook_result.attn[1] is not None
        assert hook_result.attn[1].attn_weights is not None
        print(attn_weights[0, 0])
        print(hook_result.attn[1].attn_weights[0, 0])
        print(attn_weights.shape)
        print(hook_result.attn[1].attn_weights.shape)

        torch.testing.assert_close(
            attn_weights[:1], hook_result.attn[1].attn_weights[:1], atol=1e-2, rtol=1e-2
        )

    else:
        attn_weights = (
            reconstruct_attn_weight_qk_combined_norope(
                hidden_states=hidden_states,
                q_proj_module=q_proj_module,
                k_proj_module=k_proj_module,
                num_attention_heads=num_attention_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
            )
            .detach()
            .cpu()
        )
        assert hook_result.attn is not None
        assert hook_result.attn[1] is not None
        torch.testing.assert_close(
            attn_weights, hook_result.attn[1].attn_weights, atol=1e-4, rtol=1e-4
        )


def test_reconstruct_qkv_vectors_applies_norm_module():
    """norm_module (e.g. Qwen3/Gemma3's q_norm/k_norm) must be applied over
    head_dim, before the heads/sequence transpose."""
    torch.manual_seed(0)
    hidden_size = 8
    num_heads = 2
    head_dim = 4
    q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim)
    norm = torch.nn.LayerNorm(head_dim)
    hidden_states = torch.randn(1, 3, hidden_size)

    with torch.no_grad():
        reconstructed = reconstruct_qkv_vectors(
            hidden_states=hidden_states,
            qkv_proj_module=q_proj,
            num_attention_heads=num_heads,
            module_type="q_proj",
            norm_module=norm,
        )
        expected = norm(
            q_proj(hidden_states).view(1, 3, num_heads, head_dim)
        ).transpose(1, 2)

    torch.testing.assert_close(reconstructed, expected)


def test_precompute_qk_weights_rejects_qk_norm_architectures():
    """The weight-fusion path can't represent a nonlinear q_norm/k_norm, so it
    must fail loudly instead of silently returning a wrong result."""
    architecture = BaseModelArchitecture(attn_q_norm_field="q_norm")
    q_proj = torch.nn.Linear(8, 8)
    k_proj = torch.nn.Linear(8, 8)

    with pytest.raises(NotImplementedError, match="attn_q_norm_field"):
        _precompute_qk_weights(
            q_proj_module=q_proj,
            k_proj_module=k_proj,
            num_attention_heads=2,
            head_dim=4,
            num_kv_heads=2,
            architecture=architecture,
        )


def _causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    seq_len = scores.shape[-1]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


@pytest.mark.parametrize("k_has_bias", [True, False])
@pytest.mark.parametrize("q_has_bias", [True, False])
def test_reconstruct_attn_weight_qk_combined_norope_matches_brute_force(
    q_has_bias, k_has_bias
):
    """_precompute_qk_weights's bias terms (key_side/query_side/constant, see
    QKBiasTerms) must reproduce (W_q x + b_q) . (W_k x + b_k) exactly, for
    every combination of q_proj/k_proj having a bias or not."""
    torch.manual_seed(0)
    batch, seq_len, num_heads, head_dim = 2, 5, 3, 4
    hidden_size = num_heads * head_dim

    q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=q_has_bias)
    k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=k_has_bias)
    hidden_states = torch.randn(batch, seq_len, hidden_size)

    with torch.no_grad():
        attn_weights = reconstruct_attn_weight_qk_combined_norope(
            hidden_states=hidden_states,
            q_proj_module=q_proj,
            k_proj_module=k_proj,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_heads,
        )

        query = (
            q_proj(hidden_states)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            k_proj(hidden_states)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        expected_scores = torch.einsum("bhid,bhjd->bhij", query, key) / math.sqrt(
            head_dim
        )
        expected = _causal_softmax(expected_scores)

    torch.testing.assert_close(attn_weights, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("k_has_bias", [True, False])
@pytest.mark.parametrize("q_has_bias", [True, False])
def test_reconstruct_attn_weight_qk_combined_with_rope_matches_brute_force(
    q_has_bias, k_has_bias
):
    """Same as the norope version, but RoPE-rotated: q_proj/k_proj bias is
    rotated together with the projection before the dot product, so the
    (i, j)-dependent QKBiasTerms must account for the relative rotation."""
    torch.manual_seed(0)
    batch, seq_len, num_heads, head_dim = 2, 5, 3, 4
    hidden_size = num_heads * head_dim

    q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=q_has_bias)
    k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=k_has_bias)
    hidden_states = torch.randn(batch, seq_len, hidden_size)

    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    rope_module = SimplifiedRoPEV1(inv_freq=inv_freq)
    position_embeddings = rope_module.create_position_embeddings(
        sequence_length=seq_len
    )

    with torch.no_grad():
        qk_weight_combined, qk_bias_terms = _precompute_qk_weights(
            q_proj_module=q_proj,
            k_proj_module=k_proj,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_heads,
            rope_module=rope_module,
            sequence_length=seq_len,
        )
        attn_weights = reconstruct_attn_weight_qk_combined_with_rope(
            hidden_states=hidden_states,
            qk_weight_combined=qk_weight_combined,
            qk_bias_terms=qk_bias_terms,
            head_dim=head_dim,
        )

        query = (
            q_proj(hidden_states)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            k_proj(hidden_states)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        query_roped, key_roped = _apply_rope(query, key, position_embeddings)
        expected_scores = torch.einsum(
            "bhid,bhjd->bhij", query_roped, key_roped
        ) / math.sqrt(head_dim)
        expected = _causal_softmax(expected_scores)

    torch.testing.assert_close(attn_weights, expected, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_gemma3_sandwich_norm_layer_reconstruction():
    """Gemma3's layer isn't pre-norm-only: input_layernorm -> attn ->
    post_attention_layernorm -> +residual -> pre_feedforward_layernorm -> mlp
    -> post_feedforward_layernorm -> +residual. Verify the new architecture
    fields identify the right modules in the right order by reassembling a
    full layer from existing reconstruction pieces (+ the real mlp/norm
    modules) and comparing against the model's actual layer output.
    """
    model_name = "google/gemma-3-1b-pt"
    architecture = get_model_architecture(model_name)
    assert architecture.post_attn_ln_field is not None
    assert architecture.pre_mlp_ln_field is not None
    assert architecture.post_mlp_ln_field is not None

    model_config = AutoConfig.from_pretrained(model_name)
    extractor = FeatureExtractor(model_name_or_path=model_name)
    # gemma-3-1b-pt's checkpoint is bfloat16. Reassembling a full layer chains
    # attn + 4 norms + mlp, and bf16's ~3 significant digits combined with
    # Gemma3's sharper (query_pre_attn_scalar-scaled) attention is enough to
    # push a handful of elements past a 1e-2 tolerance, even though the
    # reconstruction is algebraically correct (verified in fp32, where the
    # same reconstruction matches to ~1e-5). Run this test in fp32 so it
    # actually checks the field wiring/ordering, not bf16 rounding noise.
    extractor.model = extractor.model.float()
    extractor.configure(
        FeatureConfig.from_str(
            [
                "layers.layer_00.output",
                "layers.layer_01.output",
                "attn.layer_01.attn_weights",
            ]
        )
    )
    dataset = _create_dataset()
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=create_collator(extractor.tokenizer),
    )
    _, hook_result = next(extractor.extract_features(dataloader))

    assert hook_result.layers is not None
    layer_input = hook_result.layers[0].output
    layer_output = hook_result.layers[1].output
    assert hook_result.attn is not None
    attn_weights = hook_result.attn[1].attn_weights

    model = extractor.model
    model_module = getattr(model, architecture.model_field)
    layer_module = getattr(model_module, architecture.layers_field)[1]

    pre_attn_ln = getattr(layer_module, architecture.pre_attn_ln_field).eval()
    post_attn_ln = getattr(layer_module, architecture.post_attn_ln_field).eval()
    pre_mlp_ln = getattr(layer_module, architecture.pre_mlp_ln_field).eval()
    post_mlp_ln = getattr(layer_module, architecture.post_mlp_ln_field).eval()
    mlp_module = getattr(layer_module, architecture.mlp_field).eval()

    v_proj_module = get_v_proj_module(
        architecture=architecture, layer_index=1, model=model
    ).eval()
    o_proj_module = get_o_proj_module(
        architecture=architecture, layer_index=1, model=model
    ).eval()

    device = pre_attn_ln.weight.device
    hidden_states = layer_input.to(device)
    normed_hidden_states = pre_attn_ln(hidden_states)

    attn_output = reconstruct_attn_output_vo_combined(
        attn_weights=attn_weights.to(device),
        hidden_states=normed_hidden_states,
        v_proj_module=v_proj_module,
        o_proj_module=o_proj_module,
        num_attention_heads=get_num_attn_heads(
            model_config=model_config, architecture=architecture
        ),
        num_kv_heads=get_num_kv_heads(
            model_config=model_config, architecture=architecture
        ),
    )

    residual = hidden_states
    hidden_states = post_attn_ln(attn_output)
    hidden_states = residual + hidden_states

    residual = hidden_states
    mlp_output = mlp_module(pre_mlp_ln(hidden_states))
    mlp_output = post_mlp_ln(mlp_output)
    reconstructed_output = residual + mlp_output

    torch.testing.assert_close(
        reconstructed_output.cpu(), layer_output, atol=1e-2, rtol=1e-2
    )
