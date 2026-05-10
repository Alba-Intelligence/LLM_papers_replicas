# Julia Model-Family Workspace

This site documents the shared Julia workspace for:

- `TransformerCore.jl` — reusable tensor, schedule, and runtime primitives,
- `OpenMythos.jl` — the recurrent-depth OpenMythos model family,
- `DeepSeekv4.jl` — the architecture-first DeepSeek V4 model family,
- `OLMo.jl` — the first dense decoder family package from the comparison-driven expansion track.

The repository also keeps a narrative wiki in `docs/wiki/`. This Documenter site
is the API-oriented companion: it focuses on package entrypoints, training and
runtime surfaces, and the public types/functions exposed by each package.

## Package map

```text
TransformerCore.jl
  -> shared tensor helpers
  -> RMSNorm / RoPE
  -> training schedules and batching helpers
  -> KV-cache envelope and buffer utilities

OpenMythos.jl
  -> prelude -> recurrent block -> coda model stack
  -> ACT, LoRA, LTI injection
  -> tokenizer bridge
  -> bootstrap training and cache-aware generation

DeepSeekv4.jl
  -> CSA / HCA hybrid attention
  -> manifold hyper connections
  -> DeepSeek-style MoE routing and MTP heads
  -> bootstrap training and cache-aware generation

OLMo.jl
  -> OLMo 2-style dense decoder blocks
  -> QK-Norm MHA and inside-residual post-norm
  -> bootstrap training and cache-aware generation
```

## Reading order

1. **Manual** for the current workflow and package boundaries.
2. **TransformerCore API** for shared primitives used by the family packages.
3. **OpenMythos API** for the recurrent model family.
4. **DeepSeek V4 API** for the architecture-first DeepSeek implementation.
5. **OLMo API** for the current dense decoder-family surface.
