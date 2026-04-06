import pytest
import torch
from torch.utils.data import DataLoader

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models import SUPPORTED_MODELS
from feature_extractor.reconstruction import reconstruct_mlp_output


def _create_feature_config(tmp_path):
    return FeatureConfig(
        feature_names=[
            "mlp.layer_00.output",
            "mlp.layer_00.down_proj_input",
        ],
        output_dir=str(tmp_path),
        save_format="pt",
        batch_size=16,
    )


def _create_dataset():
    return TextDataset(
        data=[
            TextDataEntry(idx="0", text="Hello, world!"),
            TextDataEntry(idx="1", text="Testing MLP reconstruction."),
        ]
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_mlp_output_reconstruction_accuracy(model_name, tmp_path):
    config = _create_feature_config(tmp_path)
    extractor = FeatureExtractor(model_name_or_path=model_name, feature_cfg=config)

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
    assert hook_result.mlp is not None

    mlp_result = hook_result.mlp[0]
    assert mlp_result is not None
    assert mlp_result.output is not None
    assert mlp_result.down_proj_input is not None

    reconstructed = reconstruct_mlp_output(
        down_proj_input=mlp_result.down_proj_input,
        model=extractor.model,
        layer_index=0,
        architecture=extractor.architecture,
    )

    assert torch.allclose(
        reconstructed.to(dtype=mlp_result.output.dtype),
        mlp_result.output,
        rtol=1e-4,
        atol=1e-4,
    ), (
        f"Reconstructed MLP output does not match extracted output for {model_name}. "
        f"Max diff: {(reconstructed.to(dtype=mlp_result.output.dtype) - mlp_result.output).abs().max()}"
    )
