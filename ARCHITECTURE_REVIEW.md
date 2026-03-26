# Architecture Review (Updated)

## Scope

This review re-checks the current codebase with a focus on:

1. design health,
2. ease of adding new model families,
3. readability/maintainability,
4. whether recent GPT-2 architecture updates are directionally correct.

## Executive Summary

Overall status is now **improved** compared to the earlier snapshot.

- ✅ **Done**: architecture registry introduced; architecture contract tests added; capability flags added; extractor now fails fast when unsupported hooks are requested.
- ✅ **Done**: GPT-2 architecture mapping now reflects GPT-2 internals better (`attn`, `c_attn`, `c_proj`, `c_fc`, conv1d-style QKV).
- ⚠️ **Still open**: feature schema supports only `embeddings` and `layers.layer_XX.output`; README and extension guide remain minimal.

Verdict: the project is now in a much better state for adding new model families with lower regression risk.

## Current-State Evaluation

### 1) Design Guidelines / Architecture Health

#### Positive

- Model architecture resolution moved to a registry-based mechanism (`ARCHITECTURE_REGISTRY` + `resolve_model_architecture`), which is cleaner than hardcoded `if/elif` chains.
- `get_model_architecture` now accepts model instance, class, or class-name string, improving call-site ergonomics.
- Capability flags (`supports_layer_output`, `supports_attention_qkv`, `supports_mlp_output`) make architecture intent explicit.
- `FeatureExtractor` now validates hook support before installation (fail-fast behavior).

#### Remaining concerns

- Registry matchers still rely on class-name substring checks. This is acceptable short-term but still somewhat brittle for wrappers/custom subclasses.
- Capability flags exist, but only layer hooks are currently wired end-to-end in extractor/hook manager flow.

### 2) New-Model Onboarding

Current effort level: **low-to-medium** (improved from medium).

A new architecture now mostly requires:

1. adding a new `*Architecture` dataclass,
2. adding one registry entry,
3. optionally setting capability flags,
4. ensuring contract tests pass.

This is a good extensibility baseline.

### 3) GPT-2 Check (Requested)

The current GPT-2 mapping looks significantly better aligned with canonical GPT-2 blocks:

- block path: `transformer.h`
- attention module: `attn`
- combined QKV projection: `c_attn`
- attention output projection: `c_proj`
- MLP projections: `c_fc` -> `c_proj`

This is the expected direction for HF GPT-2 internals and is a clear improvement over earlier Llama-like field assumptions.

### 4) Readability / Maintainability

Readability score: **8/10** (up from 7/10).

Why improved:

- clearer architecture resolution API,
- explicit capability semantics,
- stronger tests that validate architecture contracts against real model objects.

Still worth improving:

- add a short architecture extension guide to README,
- document hook result lifecycle semantics (overwrite vs accumulation),
- consider splitting heavyweight model-loading contract tests into optional markers (e.g. `@pytest.mark.integration`).

## Updated Priority Actions

### P1 (recommended next)

- Expand feature schema grammar to match declared architecture capabilities (if attention/MLP hooks are intended user-facing).
- Add contributor documentation: "How to add a new architecture" checklist.

### P2 (polish)

- Improve registry matcher robustness (e.g., class-based predicates, or model config type keys).
- Separate fast unit tests vs slow integration/model-download tests with pytest markers.

## Final Assessment

The requested architectural improvements (registry, contract tests, capability flags) are now in place and materially improve extensibility and safety. GPT-2 mapping changes also look appropriate. The main remaining work is productization/documentation and feature-surface consistency.
