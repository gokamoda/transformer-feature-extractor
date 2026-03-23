from typing import Literal

class ModelArchitecture:
    model_field: str = "model"
    layer_field: str = "layer"
    attn_field: str = "attn"
    mlp_field: str = "mlp"
    qkv_implementation: Literal['conv1d', 'independent_linear'] = 'independent_linear'
    mlp_implementation: Literal['standard', 'gated'] = 'standard'
    
    