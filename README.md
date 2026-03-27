

# Usage

- Install dependencies:
    ```bash
    make install
    ```
- Run the greet command:
    ```bash
    make greet
    ```

- Run greet with overrides:
    ```bash
    uv run greet --config configs/debug.yaml --override debug.message=hello
    ```

## Feature names

`FeatureConfig.feature_names` では以下の形式を利用できます。

- `embeddings`
- `layers.layer_<index>.output` (例: `layers.layer_0.output`, `layers.layer_12.output`)
- `attn.layer_<index>.query`
- `attn.layer_<index>.key`
- `attn.layer_<index>.value`
- `attn.layer_<index>.attn_weights`
- `attn.layer_<index>.output`
- `mlp.layer_<index>.activation`
- `mlp.layer_<index>.output`

## 新しいモデルアーキテクチャの追加手順

1. `src/feature_extractor/models/` に `<model>.py` を追加し、`BaseModelArchitecture` を継承した dataclass を作成する。
2. モデル内部の field 名（`model_field`, `layers_field`, `attn_field` など）と QKV 実装方式を定義する。
3. `src/feature_extractor/models/__init__.py` の `ARCHITECTURE_REGISTRY` に matcher/factory を登録する。
4. `src/feature_extractor/models/architecture_test.py` の contract テストが通ることを確認する。
