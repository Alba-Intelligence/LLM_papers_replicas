# DeepSeek V4 architecture

## Current package

`DeepSeekv4.jl/` is the new Julia package for the architecture-first DeepSeek V4 implementation.

Its current surface focuses on a readable tiny-config path rather than production-scale serving or distributed training.

## Source hierarchy

The official DeepSeek V4 technical note remains the primary source for this package.

Secondary materials such as the Kili Technology DeepSeek V4 article, the Engram paper/repo, the mHC paper, and the DeepSeek-V3 report are still useful, but for a different reason:

- they help explain the research lineage around DeepSeek's late-2025 / early-2026 work,
- they clarify how DeepSeek thinks about data curation, long-context runtime, and stability at scale,
- they can inform future Julia experiments.

But they should **not** override the official V4 note when the sources disagree. In particular, the Kili article is explicitly pre-release and speculative about which ideas made it into the shipped V4 architecture.

## Implemented package surface

- `DeepSeekV4Config`
- `DeepSeekV4Model`
- `DeepSeekV4Block`
- `CompressedSparseAttention`
- `HeavilyCompressedAttention`
- `ManifoldHyperConnections`
- `HashMoEFFN`
- `MoEFFN`
- `Engram`
- `build_engram_token_lookup`
- `mtp_logits`
- `generate`
- `chunked_prefill`
- tiny and paper-inspired variant constructors

## Architectural interpretation

The package currently models the paper at the level most useful for a first Julia port:

- **CSA + HCA hybrid attention**
  - token compression,
  - sparse compressed-block selection for CSA,
  - heavy dense compression for HCA,
  - sliding-window local branch,
  - grouped output projection,
  - partial RoPE on suffix dimensions.
- **mHC residual structure**
  - expanded residual streams,
  - dynamic input/residual/output mappings,
  - Sinkhorn projection for the residual mixing matrix.
- **DeepSeek-style MoE path**
  - routed experts,
  - shared experts,
  - hash-routed early layers.
- **MTP surface**
  - multiple prediction heads exposed through `mtp_logits`.
- **Optional Engram branch**
  - deterministic local n-gram hashing at selected layers,
  - per-head static memory lookup,
  - stream-wise gating against the current mHC residual state,
  - short causal convolution over the memory value,
  - residual fusion before the attention/MoE block body.

The official note still remains the implementation authority, so Engram is kept
**explicit and gated** rather than silently assumed as core V4 parity. The
current Julia package now includes that gated branch at selected layers, with a
dependency-free compressed-token lookup helper for the first upgrade step beyond
raw token-id hashing.

## Runtime surface

The package now has a first reusable long-context runtime surface shared with `OpenMythos.jl`:

- `chunked_prefill(model, ids; chunk_size=...)`,
- `generate(model, ids; envelope=...)`,
- `TransformerCore.KVCacheEnvelope`,
- `save_kv_cache` and `load_kv_cache`.

This keeps the outer runtime contract aligned across both model families while leaving the internal CSA/HCA cache payloads family-specific. The shared envelope now also supports capacity hints so DeepSeek cache buffers can reserve decode space ahead of time without changing the inner compressed-cache layout.

## Explicitly deferred

The current package does **not** yet attempt full V4 systems parity. These remain later phases:

- Muon and hybrid ZeRO,
- FP4 quantization-aware training,
- contextual parallelism,
- deterministic fused kernels,
- paged or production-scale KV cache internals,
- million-token production serving.

Additional research branches suggested by secondary references also remain deferred unless later sources justify them directly:

- fuller Engram tokenizer-compression and host-memory lookup systems beyond the current gated in-model branch,
- explicit data-path separation between knowledge-heavy and reasoning-heavy training flows,
- speculative DeepSeek Sparse Attention variants inferred from leaks or commentary rather than official documentation.

## Training surface

`DeepSeekv4.jl/` now also includes a first trainable bootstrap milestone:

- `bootstrap_deepseek_training_config`
- `LuxHeadOnlyDeepSeekV4`
- `DeepSeekHeadTrainerState`
- `train_deepseek_head_only_step!`
- `train_deepseek_head_only!`
- `save_deepseek_checkpoint`
- `load_deepseek_checkpoint`
- `scripts/train_deepseek_tiny.jl`

This surface intentionally mirrors the OpenMythos bootstrap trainer:

- the model body stays frozen,
- only the LM head is updated,
- shared schedule, batching, checkpoint discovery, and head-loss math come from `TransformerCore.jl`,
- the default script uses local byte-encoded batches for a simple smoke-training path.

The package also now has a tiny full-model bootstrap trainer that updates the
main LM path, the current MTP heads, and optional Engram parameters on very
small configs. Broader config coverage, distributed training, and paper-scale
optimizer/runtime work remain later phases.

## Why the package is split from OpenMythos

OpenMythos is a recurrent-depth model family with a `prelude -> recurrent -> coda` core.

DeepSeek V4 is a different model family:

- non-recurrent stack,
- hybrid compressed attention,
- mHC residual mixing,
- MTP-first output surface.

Putting it in its own package keeps both implementations coherent while still allowing shared primitives through `TransformerCore.jl/`.
