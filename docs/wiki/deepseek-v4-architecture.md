# DeepSeek V4 architecture

## Current package

`DeepSeekv4.jl/` is the new Julia package for the architecture-first DeepSeek V4 implementation.

Its current surface focuses on a readable tiny-config path rather than production-scale serving or distributed training.

## Implemented package surface

- `DeepSeekV4Config`
- `DeepSeekV4Model`
- `DeepSeekV4Block`
- `CompressedSparseAttention`
- `HeavilyCompressedAttention`
- `ManifoldHyperConnections`
- `HashMoEFFN`
- `MoEFFN`
- `mtp_logits`
- `generate`
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

## Explicitly deferred

The current package does **not** yet attempt full V4 systems parity. These remain later phases:

- Muon and hybrid ZeRO,
- FP4 quantization-aware training,
- contextual parallelism,
- on-disk KV cache reuse,
- deterministic fused kernels,
- million-token production serving.

## Why the package is split from OpenMythos

OpenMythos is a recurrent-depth model family with a `prelude -> recurrent -> coda` core.

DeepSeek V4 is a different model family:

- non-recurrent stack,
- hybrid compressed attention,
- mHC residual mixing,
- MTP-first output surface.

Putting it in its own package keeps both implementations coherent while still allowing shared primitives through `TransformerCore.jl/`.
