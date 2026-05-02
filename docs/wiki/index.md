# OpenMythos Julia Wiki

This wiki is the narrative spine for the Julia reimplementation of the Python reference in `reference/OpenMythos`.

It is intentionally concept-first: the goal is to explain what the Python project is doing, what parts matter for the port, and how the Julia codebase should grow without forcing readers to reverse-engineer `main.py` from scratch.

## Current repository state

- The root repository is now a multi-package workspace.
- `OpenMythos.jl/` contains the recurrent OpenMythos Julia package with its own `Project.toml`, `src/`, `test/`, and `Manifest.toml`.
- `DeepSeekv4.jl/` contains the new DeepSeek V4 Julia package.
- `TransformerCore.jl/` contains reusable feature-last tensor helpers, `RMSNorm`, RoPE utilities, shared training helpers, and the first shared KV-cache envelope utilities.
- The authoritative implementation today is the Python reference at `reference/OpenMythos`.
- The implemented Julia slice now covers the OpenMythos primitive layer and main model stack, plus an architecture-first DeepSeek V4 package with tiny-config CSA/HCA, mHC, MTP, generation smoke paths, and bootstrap training surfaces that now include both head-only trainers and a first dense full-model OpenMythos slice.
- The first two advanced-systems runtime slices are now in place: shared cache envelopes plus lower-allocation buffer-backed cache internals behind the same runtime API.
- Future work now focuses on deeper full-model training, paged/preallocated serving internals, and distributed/performance work rather than missing repository basics.

## Reading order

1. [Usage](usage.md) - how to actually run the current Julia package.
2. [Architecture](architecture.md) - the model we are actually porting.
3. [Python reference map](python-reference-map.md) - which Python OpenMythos files matter and how they translate into Julia work.
4. [DeepSeek V4 reference map](deepseek-v4-reference-map.md) - which external DeepSeek materials map to which Julia files.
5. [References](references.md) - papers, datasets, and implementation references now used across both model families.
6. [DeepSeek V4 architecture](deepseek-v4-architecture.md) - the current DeepSeek V4 package surface and its deliberate deferrals.
7. [Multi-package workspace](multi-model-repo-plan.md) - how the Julia workspace is split across packages.
8. [Julia reimplementation plan](julia-reimplementation-plan.md) - current status and the remaining engineering phases.

There is also a small Pluto notebook example at `OpenMythos.jl/notebooks/small_example.jl`.

## Big picture

The main OpenMythos path is a Recurrent-Depth Transformer:

```text
tokens
  -> embedding
  -> prelude blocks
  -> recurrent block looped T times
  -> coda blocks
  -> norm + lm head
```

The recurrent block is the center of gravity. It combines:

- shared transformer weights across loop iterations,
- stable input injection,
- ACT halting,
- depth-wise LoRA,
- MoE feed-forward routing,
- switchable GQA or MLA attention.

That combination is what the Julia port should preserve first. Everything else is secondary.

## Scope guidance

- Treat `reference/OpenMythos/open_mythos/main.py` as the behavioral core.
- Treat `reference/OpenMythos/docs/open_mythos.md` as the clearest architecture narrative.
- Treat `reference/OpenMythos/open_mythos/moda.py` as a secondary experimental branch, not the first porting target.
- Port invariants before porting scale.
- Keep the wiki shared across packages even as Julia implementation code moves into package subdirectories.

## Maintenance rule

After a task is complete, update the relevant wiki pages if the task changed implementation status, architecture understanding, or workflow guidance, then record the work in git with a commit.

## Training status

The Julia replica now has a real bootstrap training path:

- token chunking and next-token batch construction,
- warmup + cosine learning-rate scheduling,
- resumable checkpointing,
- a small Julia `OpenMythos.jl/scripts/train_3b_fineweb_edu.jl` entrypoint,
- local-text smoke training and an optional FineWeb-Edu Python streaming bridge,
- a `LuxHeadOnlyOpenMythos` layer backed by `Optimisers.AdamW`,
- a first dense OpenMythos full-model trainer/checkpoint path for tiny GQA single-expert configs,
- a parallel `LuxHeadOnlyDeepSeekV4` bootstrap path and `DeepSeekv4.jl/scripts/train_deepseek_tiny.jl`,
- shared schedule, batching, checkpoint discovery, and head-loss helpers in `TransformerCore.jl`.

That training surface is still intentionally constrained: the new full-model slice is OpenMythos-only and dense-config-only, while DeepSeek and sparse/full generalization remain later work.

## Runtime status

The workspace now also has a first reusable long-context runtime seam:

- `TransformerCore.KVCacheEnvelope` to hold a mutable cache dictionary plus `start_pos`,
- `save_kv_cache` and `load_kv_cache` for serialized cache reuse,
- `chunked_prefill` in both `OpenMythos.jl` and `DeepSeekv4.jl`,
- envelope-aware `generate` methods that can resume from a prefetched prompt state,
- growable buffer-backed cache entries that avoid full-tensor concatenation on every append,
- envelope-level capacity hints that let both model families preallocate cache buffers before decode growth begins.

This is still a lightweight reference runtime. It now includes preallocated buffer capacity hints, but it does not yet include true paged attention or production-scale cache management.

## Future work

The training foundation is now in place. The main remaining directions are:

- full-model gradient-based training beyond the head-only Lux layer,
- broader sparse/full-model training beyond the current dense OpenMythos slice,
- paged/preallocated cache internals and longer-context serving work,
- distributed training/runtime behavior,
- deeper performance work,
- optional future documentation/API expansion as those land.
