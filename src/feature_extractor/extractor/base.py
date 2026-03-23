from feature_extractor.configs.schema import FeatureConfig
from feature_extractor.models.load import load_causal_model, load_tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader


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
        self.device = self.model.device
        self.feature_cfg = feature_cfg


    def extract_features(
        self,
        data_loader: DataLoader,
    ):
        
        