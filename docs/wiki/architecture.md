# Architecture

## Workspace-level view

The repository now carries two model families plus a shared primitive layer:

```text
TransformerCore.jl
  -> generic tensor helpers
  -> RMSNorm
  -> RoPE

OpenMythos.jl
  -> recurrent-depth language model

DeepSeekv4.jl
  -> architecture-first DeepSeek V4 stack
```

The shared wiki stays at the repository root because the important design questions span package boundaries.

## OpenMythos architecture

The authoritative OpenMythos implementation still lives in `reference/OpenMythos/open_mythos/main.py`.

It builds a decoder-only language model around a looped middle block:

```text
Input IDs
  -> token embedding
  -> Prelude: standard transformer blocks, run once
  -> Recurrent Block: one transformer block reused for multiple loop steps
  -> Coda: standard transformer blocks, run once
  -> RMSNorm
  -> LM head
```

The key invariant is that the encoded prelude output `e` is frozen and injected into every recurrent step. The recurrent hidden state is not left to drift on its own.

### OpenMythos main components

#### 1. `MythosConfig`

`MythosConfig` carries the main architecture choices:

- model width and sequence limits,
- recurrent loop count,
- attention backend (`"gqa"` or `"mla"`),
- MoE routing dimensions,
- ACT threshold,
- LoRA rank,
- RoPE settings.

#### 2. Prelude and Coda

Prelude and Coda are standard pre-norm transformer stacks that run once. They use:

- RMSNorm,
- the selected attention backend,
- dense SwiGLU-style FFNs.

#### 3. Recurrent Block

The recurrent block is the architectural center of gravity.

Per loop iteration, the model:

1. injects a loop-index embedding into the hidden state,
2. combines hidden state with the frozen encoded input `e`,
3. applies one transformer block with MoE FFN,
4. adds a depth-wise LoRA delta,
5. updates the state with an LTI-stable injection rule,
6. computes ACT halting probabilities and accumulates a weighted output.

This is why OpenMythos remains its own package rather than being folded into a generic transformer stack.

#### 4. Attention backends

| Backend | Julia type | Role |
| --- | --- | --- |
| GQA | `GQAttention` | simpler baseline path |
| MLA | `MLAttention` | higher-priority parity path |

Both use RoPE and KV caching, but the cache structure differs:

- GQA caches full K and V tensors with fewer KV heads than Q heads.
- MLA caches a compressed latent representation and reconstructs parts of K and V on demand.

## DeepSeek V4 architecture

`DeepSeekv4.jl/` is a separate package because DeepSeek V4 is a different model family, not an OpenMythos variant.

The current Julia package models the paper at the architecture-first level:

```text
Input IDs
  -> token embedding
  -> non-recurrent DeepSeek block stack
       -> CSA / HCA hybrid attention
       -> mHC residual mixing
       -> routed/shared/hash MoE FFN
  -> readout mixing
  -> RMSNorm
  -> LM head + MTP heads
```

### DeepSeek-specific components

- `DeepSeekV4Config`
- `CompressedSparseAttention`
- `HeavilyCompressedAttention`
- `ManifoldHyperConnections`
- `HashMoEFFN`
- `DeepSeekV4Block`
- `DeepSeekV4Model`
- `mtp_logits`

The current implementation is intentionally correctness-first and tiny-config-first. It is not yet a production-scale training or serving system.

## Shared primitive layer

`TransformerCore.jl/` currently holds the code that is obviously reusable across both packages:

- feature-last tensor helpers,
- embedding and categorical sampling helpers,
- row/column softmax helpers,
- `RMSNorm`,
- RoPE precomputation and application,
- shared schedule/batching/checkpoint-discovery/head-loss helpers,
- `KVCacheEnvelope`, `save_kv_cache`, `load_kv_cache`,
- growable axis-append cache buffers used by both model families.

That package should remain architecture-agnostic. Recurrence, CSA/HCA, mHC, training wrappers, and routing policies stay in the model-family packages until a genuinely stable shared abstraction exists.

## Runtime foundations

The first cross-package long-context runtime seam is now shared rather than model-specific:

- `TransformerCore.KVCacheEnvelope` wraps a mutable `Dict{String, Any}` cache plus the current cached prefix length,
- both model families expose `chunked_prefill` to build that envelope in prompt-sized chunks,
- both `generate` entrypoints can continue from an existing envelope instead of rebuilding the prompt cache from scratch,
- cache growth now uses shared growable axis buffers instead of repeated full `cat(...; dims=2)` copies.

The underlying cache payloads are still family-specific:

- OpenMythos GQA stores full K/V tensors, while MLA stores compressed latent cache pieces,
- DeepSeek CSA and HCA store their own compressed attention state,
- those payloads now live inside buffer-backed cache entries so decode/prefill appends reuse storage rather than reallocating the whole prefix every step.

That is intentional. The shared abstraction is the outer runtime contract plus the reusable growth primitive, not a forced uniform inner cache layout.

### OpenMythos runtime note

OpenMythos now uses a prefix-aware causal mask during cached multi-token prefill. That keeps chunked prefill behavior aligned with ordinary full-prompt generation even when a later chunk attends over an already-populated cache prefix.

## Why the tests matter

The strongest implementation cues are still invariant tests rather than scale-oriented training scripts.

For OpenMythos, `reference/OpenMythos/tests/test_main.py` describes the minimum behavioral bar:

- RMSNorm shape and RMS properties,
- RoPE shape, norm preservation, and relative-position behavior,
- cache behavior,
- halting logic,
- LoRA depth handling,
- recurrent block output shape and stability assumptions.

For DeepSeek V4, the Julia package currently uses tiny-config invariants:

- forward logits shape,
- MTP output shape,
- cache smoke behavior,
- generation smoke behavior.

## What remains primary vs deferred

### Primary OpenMythos parity target

- `open_mythos/main.py`
- `open_mythos/tokenizer.py`
- `open_mythos/variants.py`
- `tests/test_main.py`
- `tests/test_tokenizer.py`
- `docs/open_mythos.md`

### Primary DeepSeek V4 target

- the DeepSeek V4 technical note,
- the introduction video,
- the current `DeepSeekv4.jl/` source and tests.

### Explicit deferrals

- `open_mythos/moda.py`
- benchmark scripts in `reference/OpenMythos/tests/`
- Muon and hybrid ZeRO
- FP4 QAT
- contextual parallelism
- production million-token serving work
- paged/preallocated KV cache internals
