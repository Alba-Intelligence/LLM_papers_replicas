# Multi-package workspace

## Goal

Keep one shared repository and wiki while allowing multiple Julia packages to evolve at different speeds.

The current workspace already contains OpenMythos- and DeepSeek-specific packages, but the broader architecture program now also follows the PDF/article inventory recorded in [Big LLM architecture comparison map](llm-architecture-comparison-map.md). That inventory is broader than the currently implemented packages, so the package split described here should be read as the current nucleus, not the final endpoint.

## Current workspace split

| Path                            | Purpose                                                                    |
| ------------------------------- | -------------------------------------------------------------------------- |
| `OpenMythos.jl/`                | recurrent OpenMythos package                                               |
| `DeepSeekv4.jl/`                | architecture-first DeepSeek V4 package                                     |
| `TransformerCore.jl/`           | shared low-level primitives plus Lux-native training/checkpoint foundation |
| `TextDataCore.jl/`              | shared tokenizer and local text-data helpers                               |
| `docs/wiki/`                    | common documentation                                                       |
| `reference/private/OpenMythos/` | vendored OpenMythos Python source of truth                                 |

## What belongs in `TransformerCore.jl/`

Only generic, reusable building blocks:

- feature-last tensor helpers,
- sampling helpers,
- `RMSNorm`,
- RoPE precomputation and application,
- warmup/cosine schedule utilities,
- generic token chunking and batching helpers,
- head-only loss/gradient math,
- Lux-native feature-last layers,
- generic next-token trainer state / train-step helpers,
- family/mode-aware checkpoint helpers.

This package should stay small and boring.

## What belongs in `TextDataCore.jl/`

Shared text/tokenizer helpers that are reusable but would make `TransformerCore.jl`
too policy-heavy or dependency-heavy:

- Julia-native GPT/tiktoken-style tokenizer wrappers,
- zero-based encode/decode APIs for training code,
- vocabulary-surface extraction helpers,
- local parquet text-column loading,
- tokenizer-driven next-token batch construction from text shards.

## What should stay package-specific

- OpenMythos recurrence, ACT, LoRA, and LTI logic,
- DeepSeek V4 CSA/HCA attention, mHC, MTP, and routing choices,
- model-family naming/policy wrappers on top of shared tokenizers,
- remote dataset bridges and package-specific dataset naming,
- package-specific training wrappers while model-specific hidden-state computation still differs.

## Upcoming shared extraction fronts

As the PDF-scoped family ports expand, the most likely reusable extraction fronts are:

- decoder scaffolds and layer-schedule helpers,
- attention-backend families,
- normalization-policy families,
- sparse FFN / MoE block families,
- linear/state-space sequence blocks,
- auxiliary decoding heads such as MTP.

These should become new shared packages only when a second concrete family implementation makes the API real.

## Rule of thumb

Do not move code into `TransformerCore.jl/` just because two packages happen to use it once.

Move code there when:

1. it is architecture-agnostic,
2. it is already used by more than one package,
3. its API is stable enough that extracting it reduces duplication rather than spreading churn.
