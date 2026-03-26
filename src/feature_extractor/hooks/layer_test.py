from feature_extractor.hooks.layer import LayerHookManager
from transformers import AutoModelForCausalLM
from feature_extractor.models.architecture import get_model_architecture
from feature_extractor.extractor.extractor_test import _create_feature_config
from feature_extractor.models import SUPPORTED_MODELS
import pytest
import torch    
from feature_extractor.models import load_tokenizer

def test_resolve_layer_indices():
    feature_config = _create_feature_config()

    layer_indices = LayerHookManager._resolve_layer_index(feature_config)
    assert layer_indices == [0], f"Expected [0], got {layer_indices}"


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_hook(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    architecture = get_model_architecture(model.__class__.__name__)
    feature_config = _create_feature_config()

    hook_manager = LayerHookManager(
        model=model,
        architecture=architecture,
        feature_cfg=feature_config
    )

    assert len(hook_manager.layer_hooks) == 1, f"Expected 1 hook, got {len(hook_manager.layer_hooks)}"

    tokenizer = load_tokenizer(model_name)

    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    assert isinstance(hook_manager.layer_hooks[0].result.hidden_states.shape, torch.Size)



    hook_manager.remove_hooks()


