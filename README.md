# feature-extractor

Hugging Face の causal LM に hook を掛けて、forward の途中の特徴（埋め込み、各層の入出力、attention の q/k/v と重み、MLP の活性など）を取り出すライブラリ。必要な最も深い層まで計算したら forward を打ち切る。

## インストール

開発するとき（このリポジトリで作業するとき）:

```bash
make install   # GPU があれば CUDA 版、なければ CPU 版の PyTorch を入れる
```

ほかのリポジトリから使うときは、GitHub から rev を固定して取得する（`pyproject.toml`）:

```toml
[project]
dependencies = ["feature-extractor"]

[tool.uv.sources]
feature-extractor = { git = "https://github.com/gokamoda/transformer-feature-extractor.git", rev = "<commit>" }
```

## 使い方

```python
from torch.utils.data import DataLoader

from feature_extractor import FeatureExtractor
from feature_extractor.configs import FeatureConfig
from feature_extractor.data.dataset import TextDataEntry, TextDataset, create_collator

extractor = FeatureExtractor("openai-community/gpt2")  # モデルと tokenizer を読み込む
extractor.configure(FeatureConfig.from_str(["attn.layer_00.attn_weights"]))

dataset = TextDataset([TextDataEntry(idx="0", text="Hello world")])
loader = DataLoader(dataset, batch_size=8, collate_fn=create_collator(extractor.tokenizer))
for batch, features in extractor.extract_features(loader):
    weights = features.attn[0].attn_weights  # [batch, head, seq, seq]
```

- `extract_features` はバッチごとに、入力のバッチと取り出した特徴（`HookResult`）を返す。
- `HookResult` は `embeddings`, `layers`, `attn`, `mlp` を持ち、`layers`, `attn`, `mlp` は層ごとのリスト（指定していない層は `None`）。
- モデルだけ、tokenizer だけが要るときは `feature_extractor.models` の `load_causal_model`, `load_tokenizer` を使う。

## tokenizer

`load_tokenizer` は [tokenizer-tools](https://github.com/gokamoda/tokenizer-tools) で読み込む。corpus-tools も同じものを使うので、頻度を数えた tokenizer とモデルを動かす tokenizer が一致する。

- tokenizer-tools が既知の問題を直す。例えば OpenAI の GPT-2 は先頭に `<|endoftext|>` を付け、rinna/japanese-gpt2-small は小文字にしてから tokenize する。詳しくは tokenizer-tools の README。
- そのうえで、バッチにまとめるために左側をパディングし、pad には EOS を使う。

## 対応モデル

アーキテクチャはモデルのクラス名から決まる（`src/feature_extractor/models/__init__.py` の `ARCHITECTURE_REGISTRY`）。

| クラス | 動作を確認したモデル（`SUPPORTED_MODELS`、テスト対象） |
|---|---|
| `GPT2LMHeadModel` | openai-community/gpt2 |
| `LlamaForCausalLM` | meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.2-1B, HuggingFaceTB/SmolLM2-135M |
| `Qwen2ForCausalLM` | Qwen/Qwen2.5-0.5B |
| `Qwen3ForCausalLM` | Qwen/Qwen3-0.6B |
| `Gemma3ForCausalLM` | google/gemma-3-1b-pt |
| `MistralForCausalLM` | （登録のみ。Llama と同じ構成） |

登録にないクラスは、既定の構成（`BaseModelArchitecture`）として扱い、警告を出す。

## 特徴名

`FeatureConfig.from_str` に渡す名前。`<i>` は層の番号で、`layer_0` と `layer_00` のどちらでもよい（取り出した結果の名前は 2 桁の `layer_00`）。

- `embeddings`
- `layers.layer_<i>.input`, `layers.layer_<i>.output`
- `attn.layer_<i>.query`, `attn.layer_<i>.key`, `attn.layer_<i>.value`（GQA のモデルでも、k と v はヘッドの数に展開する）
- `attn.layer_<i>.attn_weights`
- `attn.layer_<i>.output`
- `attn.layer_<i>.attention_mask`
- `attn.layer_<i>.positional_embedding`
- `mlp.layer_<i>.activation`
- `mlp.layer_<i>.down_proj_input`
- `mlp.layer_<i>.output`

## 新しいモデルアーキテクチャの追加手順

1. `src/feature_extractor/models/` に `<model>.py` を追加し、`BaseModelArchitecture` を継承した dataclass を作成する。
2. モデル内部の field 名（`model_field`, `layers_field`, `attn_field` など）と QKV 実装方式を定義する。
3. `src/feature_extractor/models/__init__.py` の `ARCHITECTURE_REGISTRY` に matcher/factory を登録する。
4. `SUPPORTED_MODELS` にモデルを足し、`src/feature_extractor/models/architecture_test.py` の contract テストが通ることを確認する。
