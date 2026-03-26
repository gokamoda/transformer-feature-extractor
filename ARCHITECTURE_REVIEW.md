# Architecture Review

## Scope

This review evaluates the current codebase from three perspectives:

1. Whether the design guidelines look healthy.
2. Whether adding support for new models is straightforward.
3. Whether the implementation is reasonably readable/maintainable.

## Executive Summary

Overall, the architecture is **small, understandable, and directionally good** for a hook-based feature extractor.

- **Strengths**: clear domain boundaries (`models`, `hooks`, `extractor`, `configs`, `data`), explicit model architecture abstraction, and test coverage for basic loading/hook behavior.
- **Primary risks**: architecture dispatch is class-name-string based and currently validated incorrectly in one test; GPT-2 architecture defaults appear mismatched to actual GPT-2 module internals; feature schema currently validates only a narrow subset (`embeddings`, `layers.layer_XX.output`) despite broader hook abstractions in code.
- **Verdict**: good foundation, but if you plan to scale to many architectures, introducing a model adapter registry and stronger contract tests is strongly recommended.

## Detailed Evaluation

### 1) Design Guidelines / Architecture Health

#### What is good

- The central orchestration flow in `FeatureExtractor` is compact and easy to follow: load model/tokenizer -> infer architecture -> install hooks -> iterate batches -> emit hook results.
- Hook responsibilities are separated (`Hook` base + layer-specific manager/result), which keeps feature-capture mechanics decoupled from model loading.
- A dataclass-based architecture descriptor (`BaseModelArchitecture`) is a practical pattern for mapping model-specific field names.

#### What should be improved

- The architecture decision path is string matching against class names, which is brittle for forks/wrappers and easier to break silently.
- `FeatureConfig` pattern validation currently allows only layer output and embeddings, while architecture metadata already includes attention/MLP descriptors. This creates a mismatch between intended extensibility and validated user surface area.
- `Hook` stores only the latest forward result; there is no lifecycle policy (append/clear/per-step) documented, which can be surprising for multi-batch usage.

### 2) Ease of Supporting New Models

#### Current effort level: **medium**

Adding a new model likely requires:

1. Creating a new architecture dataclass variant.
2. Updating dispatch logic in `get_model_architecture`.
3. Ensuring field names align with actual HF module internals.
4. Updating `SUPPORTED_MODELS` and tests.

This is manageable for a few architectures, but scaling will become painful without a registry.

#### Recommended direction

- Replace manual class-name conditionals with a registry table keyed by actual model class (or robust predicates).
- Add architecture-level contract tests that instantiate each supported model and assert all required fields exist.
- Separate "model family defaults" from "feature extraction capability flags" so unsupported hooks fail early with actionable errors.

### 3) Readability / Maintainability

#### Readability score: **good (7/10)**

- File/module naming is intuitive.
- Functions are short and mostly single-purpose.
- Type annotations are present in key places.

#### Readability pain points

- A few tests assert assumptions indirectly and may pass while not validating intended behavior.
- Some naming/semantics in GPT-2 architecture suggest copy-over from Llama defaults, which can confuse future maintainers.
- Public README is still minimal; operation and extension docs are not yet strong enough for external contributors.

## High-Impact Findings

1. **Architecture test mismatch**: one test passes model IDs to `get_model_architecture`, but production code expects class names.
2. **Potential GPT-2 mapping mismatch**: GPT-2 block/attention fields appear likely inconsistent with canonical GPT-2 internals.
3. **Feature schema / hook capability gap**: schema currently validates fewer feature forms than hooks/architecture suggest.

## Prioritized Action Plan

### P0 (correctness)

- Fix architecture test to pass actual model class names or instantiate models for validation.
- Audit GPT-2 architecture fields against current `transformers` GPT-2 implementation.

### P1 (extensibility)

- Introduce architecture registry pattern and architecture contract tests.
- Add explicit capability flags per architecture (e.g., supports_layer_output / supports_attn_qkv / supports_mlp).

### P2 (developer experience)

- Expand README with:
  - supported models + caveats,
  - feature-name grammar,
  - "how to add a model" checklist.
- Clarify hook result lifecycle (per-forward overwrite vs accumulation).

## Final Assessment

- **Design quality**: solid baseline for an early-stage feature extractor.
- **New model onboarding**: feasible today, but should be hardened before broad model-family expansion.
- **Readability**: generally good, with a few correctness/consistency hotspots worth addressing soon.
