from dataclasses import dataclass

from .architecture import BaseModelArchitecture


@dataclass
class LlamaArchitecture(BaseModelArchitecture):
    supports_layer_output: bool = True
    supports_attention_qkv: bool = True
    supports_mlp_output: bool = True
