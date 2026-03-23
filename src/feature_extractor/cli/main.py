from pathlib import Path

from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.extractor.base import BaseFeatureExtractor
from feature_extractor.data.dataset import Entry, TextDataset
from feature_extractor.data.load import load_jsonl_text_dataset
from torch.utils.data import DataLoader

DEFAULT_DATASET_PATH = Path("outputs/dataset/tinystories/train.jsonl")


def main(
    model_name_or_path: str,
    dataset: TextDataset,
    feature_cfg: FeatureConfig,
):
    extractor = BaseFeatureExtractor(model_name_or_path, feature_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=feature_cfg.batch_size,
        shuffle=False,
        collate_fn=dataset.make_collate_fn(extractor.tokenizer),
    )
    for result in extractor.extract_features(dataloader):
        print(result)
        break  # just do one batch for testing

if __name__ == "__main__":
    if DEFAULT_DATASET_PATH.exists():
        dataset_raw = load_jsonl_text_dataset(DEFAULT_DATASET_PATH)
        entries = [
            Entry(idx=item.get("idx", entry_index), text=item["text"])
            for entry_index, item in enumerate(dataset_raw)
        ]
    else:
        entries = [
            Entry(idx=0, text="Hello world."),
            Entry(idx=1, text="Feature extraction smoke test."),
        ]

    dataset = TextDataset(data=entries)
    main(
        model_name_or_path="openai-community/gpt2",
        dataset=dataset,
        feature_cfg=FeatureConfig(
            feature_names=[
                "embeddings",
                "residual.layer_00.pre_attn",
                "residual.layer_00.post_ffn",
            ],
            output_dir="outputs/features",
            save_format="pt",
            batch_size=8,
        )
    )
