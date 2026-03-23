from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.extractor.base import BaseFeatureExtractor
from feature_extractor.data.load import load_jsonl_text_dataset
from feature_extractor.data.dataset import Entry, TextDataset
from torch.utils.data import DataLoader



def main(
    model_name_or_path: str,
    dataset: TextDataset,
    feature_cfg: FeatureConfig,
):
    extractor = BaseFeatureExtractor(model_name_or_path, feature_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=dataset.make_collate_fn(extractor.tokenizer),
    )
    for result in extractor.extract_features(dataloader):
        print(result)
        break  # just do one batch for testing

if __name__ == "__main__":
    dataset_path = "outputs/dataset/tinystories/train.jsonl"
    dataset_raw = load_jsonl_text_dataset(dataset_path)

    dataset = TextDataset(
        data = [Entry(idx=item["idx"], text=item["text"]) for item in dataset_raw]
    )
    main(
        model_name_or_path="openai-community/gpt2",
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
                "attn.layer_00.qk_logits",
                "attn.layer_00.weights",
                "mlp.layer_00.activation",
            ],
            output_dir="outputs/features",
            save_format="pt",
            batch_size=8,
        )
    )
