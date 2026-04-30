# Multi-package workspace

## Goal

Keep one shared repository and wiki while allowing multiple Julia packages to evolve at different speeds.

## Current workspace split

| Path | Purpose |
| --- | --- |
| `OpenMythos.jl/` | recurrent OpenMythos package |
| `DeepSeekv4.jl/` | architecture-first DeepSeek V4 package |
| `TransformerCore.jl/` | shared low-level primitives and training utilities |
| `docs/wiki/` | common documentation |
| `reference/OpenMythos/` | upstream Python source of truth |

## What belongs in `TransformerCore.jl/`

Only generic, reusable building blocks:

- feature-last tensor helpers,
- sampling helpers,
- `RMSNorm`,
- RoPE precomputation and application,
- warmup/cosine schedule utilities,
- generic token chunking and batching helpers,
- head-only loss/gradient math,
- checkpoint discovery helpers.

This package should stay small and boring.

## What should stay package-specific

- OpenMythos recurrence, ACT, LoRA, and LTI logic,
- DeepSeek V4 CSA/HCA attention, mHC, MTP, and routing choices,
- package-specific training wrappers while model-specific hidden-state computation still differs.

## Rule of thumb

Do not move code into `TransformerCore.jl/` just because two packages happen to use it once.

Move code there when:

1. it is architecture-agnostic,
2. it is already used by more than one package,
3. its API is stable enough that extracting it reduces duplication rather than spreading churn.
