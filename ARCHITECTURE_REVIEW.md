# Architecture Review (2026-03-27)

## 評価の観点

このレビューは、以下を重点にコードベースを確認したものです。

1. 設計指針（責務分離・拡張ポイント・安全性）が妥当か
2. 新しいモデル系統への対応がどれだけ容易か
3. 読みやすさ・保守性が適度か

確認対象の中心は `models/`, `hooks/`, `extractor/`, `configs/` です。

## 総評

- **設計方針は概ね良好**です。特に「アーキテクチャ記述を dataclass で外出し」「レジストリ経由で解決」「Extractor 側で capability を見て fail-fast」という流れは、拡張に強い構造です。
- **新規モデル対応の難易度は low〜medium**。`*Architecture` 追加 + レジストリ登録 + 既存 contract test で最低限の担保ができるため、導入コストは抑えられています。
- **可読性は中〜高**。モジュール分割は適切で、命名も概ね一貫しています。
- 一方で、**attention hook の一部に不具合化しやすい分岐**と、**README の利用者向けガイド不足**が残っています。

## 詳細評価

### 1) 設計指針の妥当性

#### 良い点

- `BaseModelArchitecture` が「モデル内部のフィールド名差異」を吸収する主軸になっており、hook 実装の共通化に効いています。
- `ARCHITECTURE_REGISTRY` により、モデル種別の増加時でも分岐が局所化されています。
- `FeatureExtractor.install_hooks()` で `supports_*` を確認し、未対応の feature 指定時に早期失敗できるのは安全です。

#### 改善余地

- レジストリ判定が「クラス名文字列の部分一致」依存で、ラッパー経由や派生クラスでは脆い可能性があります。
  - 例: `config.model_type` や `PretrainedConfig` クラス単位での判定に寄せると堅牢化しやすいです。
- schema 上は `attn.*` が指定可能ですが、`embeddings` は validation だけで実取得の実装が見当たらず、機能面の整合性が弱いです。

### 2) 新しいモデルへの対応容易性

現在の構成なら、基本的に次で追加できます。

1. `models/<new_model>.py` に `*Architecture` を追加
2. `models/__init__.py` の `ARCHITECTURE_REGISTRY` に entry 追加
3. 必要なら `supports_*` と QKV 実装種別（独立 linear / conv1d）を指定
4. 既存の architecture contract test を通す

この導線はシンプルで、**導入ハードルは低め**です。

### 3) Readability / 保守性

#### 良い点

- `hooks/base.py` で hook 共通処理をまとめ、`layer.py` / `attention.py` で責務を分離できています。
- テストが「構成の正しさ」「shape の期待値」「モデル別挙動」をある程度カバーしています。

#### 気になった点

- `AttentionHookManager.get_features()` で、feature 指定の組み合わせ次第では `query_features` などが未初期化のまま参照されうる分岐があります（`attn.output` のみ指定など）。
- `FeatureConfig` の layer index パターンが2桁固定（`layer_\d{2}`）で、100層以上を想定したときに将来制約になります。
- README が最小限で、実運用に必要な「feature 名の仕様」「新規 architecture 追加手順」が不足しています。

## 優先改善提案

### P1（優先）

1. **AttentionHookManager.get_features の初期化ロジックを安全化**
   - `query/key/value/attn_weights/output` を常に `num_layers` 長で先に初期化する。
   - 取得対象がない場合でも空配列ではなく `None` 埋め配列で統一する。
2. **README に extension guide 追加**
   - 「新しいモデルを追加する最短手順」
   - 「feature 名 grammar と期待 tensor shape」

### P2（中期）

1. レジストリの matcher を class-name 依存から config ベースへ移行。
2. schema の layer index 正規表現を可変桁に変更。
3. 重い統合テスト（実モデルDL）は `integration` marker 分離で CI の制御性を上げる。

## 最終判断

- **設計方針は問題ない（Go 寄り）**。
- **新モデル対応も比較的容易**。
- **可読性も概ね良好**。
- ただし本番運用に向けては、`AttentionHookManager` の分岐安全性と利用者向けドキュメントの補強を先に行うのが妥当です。
