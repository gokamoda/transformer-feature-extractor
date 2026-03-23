from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.data.dataset import Entry, TextDataset
from feature_extractor.data.load import load_jsonl_text_dataset
from feature_extractor.extractor.base import BaseFeatureExtractor
from feature_extractor.hooks.results import ExtractorResult


def build_dataloader(
    dataset: TextDataset,
    extractor: BaseFeatureExtractor,
    *,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.make_collate_fn(extractor.tokenizer),
    )


def extract_first_result(
    model_name_or_path: str,
    dataset: TextDataset,
    feature_cfg: FeatureConfig,
    hook_dtype: torch.dtype = torch.float32,
) -> ExtractorResult | None:
    extractor = BaseFeatureExtractor(
        model_name_or_path,
        feature_cfg,
        hook_dtype=hook_dtype,
    )
    dataloader = build_dataloader(
        dataset,
        extractor,
        batch_size=feature_cfg.batch_size,
    )
    for result in tqdm(extractor.extract_features(dataloader)):
        return result
    return None


def run_demo() -> None:
    dataset_path = "outputs/dataset/tinystories/train.jsonl"
    dataset_raw = load_jsonl_text_dataset(dataset_path)
    dataset = TextDataset(
        [Entry(idx=item["idx"], text=item["text"]) for item in dataset_raw]
    )
    result = extract_first_result(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        dataset=dataset,
        feature_cfg=FeatureConfig(
            feature_names=[
                "embeddings",
                "layer.layer_00.attn_output",
                "layer.layer_00.ffn_output",
                "layer.layer_00.output",
                "attn.layer_00.query",
                "attn.layer_00.key",
                "attn.layer_00.value",
                "attn.layer_00.weights",
                "mlp.layer_00.activation",
            ],
            output_dir="outputs/features",
            save_format="pt",
            batch_size=2,
        ),
        hook_dtype=torch.float32,
    )
    if result is not None:
        print(result)


if __name__ == "__main__":
    run_demo()
