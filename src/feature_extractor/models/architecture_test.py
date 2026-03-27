import pytest
from transformers import AutoModelForCausalLM

from feature_extractor.models import SUPPORTED_MODELS, get_model_architecture
from feature_extractor.models.architecture import (
    QKV_IMPLEMENTATION_CONV1D,
    QKV_IMPLEMENTATION_INDEPENDENT_LINEAR,
)


def _get_attr_path(obj: object, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_supported_models_have_architectures(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    architecture = get_model_architecture(model)
    assert architecture is not None, (
        f"Model {model_name} does not have a defined architecture"
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_supported_models_satisfy_architecture_contracts(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    architecture = get_model_architecture(model)

    if "meta-llama/Llama" in model_name:
        assert architecture.__class__.__name__ == "LlamaArchitecture", (
            f"Expected LlamaArchitecture for model {model_name}, got {architecture.__class__.__name__}"
        )
        assert (
            architecture.attn_qkv_implementation
            == QKV_IMPLEMENTATION_INDEPENDENT_LINEAR
        )
    elif "gpt2" in model_name:
        assert architecture.__class__.__name__ == "GPT2Architecture", (
            f"Expected GPT2Architecture for model {model_name}, got {architecture.__class__.__name__}"
        )
        assert architecture.attn_qkv_implementation == QKV_IMPLEMENTATION_CONV1D

    model_root = _get_attr_path(model, architecture.model_field)
    layers = _get_attr_path(model_root, architecture.layers_field)

    assert len(layers) > 0, f"Model {model_name} has no layers at configured path"

    if architecture.supports_layer_output:
        assert architecture.layer_return_fields, (
            f"Model {model_name} must define layer_return_fields "
            "when supports_layer_output=True"
        )

    first_layer = layers[0]

    if architecture.supports_attention_qkv:
        attn_module = _get_attr_path(first_layer, architecture.attn_field)
        assert attn_module is not None, (
            f"Model {model_name} missing attention module at {architecture.attn_field}"
        )
        assert hasattr(attn_module, architecture.attn_o_proj_field), (
            f"Model {model_name} missing attention output projection field "
            f"{architecture.attn_o_proj_field}"
        )

    if architecture.supports_mlp_output:
        mlp_module = _get_attr_path(first_layer, architecture.mlp_field)
        assert mlp_module is not None, (
            f"Model {model_name} missing MLP module at {architecture.mlp_field}"
        )
        assert hasattr(mlp_module, architecture.mlp_down_proj_field), (
            f"Model {model_name} missing MLP down projection field "
            f"{architecture.mlp_down_proj_field}"
        )
