# Julia model-family replication wiki

This wiki is the narrative spine for the repository's Julia model-family replication workspace.

It started as documentation for the vendored OpenMythos Python reference in `reference/private/OpenMythos`, but it now also tracks the architecture-first DeepSeek work and a broader comparison-driven replication program based on `reference/private/The Big LLM Architecture Comparison.pdf` plus `reference/private/Big-LLM-Architecture-models.yml`.

It is intentionally concept-first: the goal is to explain what the active source materials are, what parts matter for the Julia implementations, and how the workspace should grow without forcing readers to reconstruct the repository history from scratch.

## Current repository state

- The root repository is now a multi-package workspace.
- `OpenMythos.jl/` contains the recurrent OpenMythos Julia package with its own `Project.toml`, `src/`, `test/`, and `Manifest.toml`.
- `DeepSeekv4.jl/` contains the new DeepSeek V4 Julia package.
- `TransformerCore.jl/` contains reusable feature-last tensor helpers, `RMSNorm`, RoPE utilities, shared Lux-native layer/trainer helpers, family/mode-aware checkpoint helpers, and the first shared KV-cache envelope utilities.
- `TextDataCore.jl/` contains shared Julia-native tokenizer and local text-data helpers, including GPT/tiktoken-style BPE tokenizers, vocabulary-surface extraction, and parquet-backed next-token batch loading.
- the workspace now also keeps a comparison-driven architecture inventory rooted in `reference/private/The Big LLM Architecture Comparison.pdf` and `reference/private/Big-LLM-Architecture-models.yml`; for that expansion track, only the PDF/article-covered families are in active scope by default.
- For `OpenMythos.jl`, the authoritative vendored Python reference today is `reference/private/OpenMythos`.
- The implemented Julia slice now covers the OpenMythos primitive layer and main model stack, plus an architecture-first DeepSeek V4 package with tiny-config CSA/HCA, mHC, an optional gated Engram branch, MTP, generation smoke paths, and bootstrap training surfaces that now include both head-only trainers, a first dense full-model OpenMythos slice, and a first tiny DeepSeek full-model slice.
- The advanced-systems runtime work now includes shared cache envelopes, lower-allocation buffer-backed cache growth, and a first true paged cache-buffer implementation beneath the same outer runtime API.
- The workspace now also has a shared `Documenter.jl` site under `docs/` in addition to the narrative wiki under `docs/wiki/`.
- Future work now focuses on deeper full-model training, paged/preallocated serving internals, and distributed/performance work rather than missing repository basics.
- The current comparison-driven family order is `OLMo.jl` first, `Gemma.jl` second, and `Qwen.jl` third, with any shared decoder or MoE extraction delayed until the code has a real second user.

## Reading order

1. [Usage](usage.md) - how to actually run the current Julia packages.
2. [Architecture](architecture.md) - the workspace architecture that already exists.
3. [Big LLM architecture comparison map](llm-architecture-comparison-map.md) - the PDF-scoped architecture inventory and shared-abstraction map for future family ports.
4. [Python reference map](python-reference-map.md) - which Python OpenMythos files matter and how they translate into Julia work.
5. [DeepSeek V4 reference map](deepseek-v4-reference-map.md) - which external DeepSeek materials map to which Julia files.
6. [References](references.md) - papers, datasets, and implementation references now used across model families.
7. [DeepSeek V4 architecture](deepseek-v4-architecture.md) - the current DeepSeek V4 package surface and its deliberate deferrals.
8. [Multi-package workspace](multi-model-repo-plan.md) - how the Julia workspace is split across packages.
9. [Julia reimplementation plan](julia-reimplementation-plan.md) - current status and the remaining engineering phases.
10. [HypergraphReasoning documentation](hypergraph-reasoning.md) - the integrated legacy Typst and specification material for the HypergraphReasoning Julia replication.

There is also a small Pluto notebook example at `notebooks/openmythos/small_example.jl`.

## Big picture

The deepest currently implemented path is still the OpenMythos recurrent-depth transformer:

```text
tokens
  -> embedding
  -> prelude blocks
  -> recurrent block looped T times
  -> coda blocks
  -> norm + lm head
```

The recurrent block is the center of gravity for the current OpenMythos package. It combines:

- shared transformer weights across loop iterations,
- stable input injection,
- ACT halting,
- depth-wise LoRA,
- MoE feed-forward routing,
- switchable GQA or MLA attention.

That combination is what the Julia OpenMythos port should preserve first.

At the workspace level, however, it is no longer the only active architecture direction. The new comparison map records the broader set of decoder families, MoE variants, norm policies, sliding-window layouts, linear/state-space hybrids, and long-context attention tricks that the repository is now expected to absorb over time.

## Scope guidance

- Treat `reference/private/OpenMythos/open_mythos/main.py` as the behavioral core for OpenMythos-specific work.
- Treat `reference/private/OpenMythos/docs/open_mythos.md` as the clearest OpenMythos architecture narrative.
- Treat `reference/private/OpenMythos/open_mythos/moda.py` as a secondary experimental branch, not the first OpenMythos porting target.
- For broader family work, use `llm-architecture-comparison-map.md` to decide scope before touching code.
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
- local-text smoke training, a shared Julia-native tokenizer/text-data package, a Julia-native FineWeb rows-api smoke path, and a first Julia-native remote parquet-shard path,
- a `LuxHeadOnlyOpenMythos` layer backed by `Optimisers.AdamW`,
- a broadened OpenMythos full-model trainer/checkpoint path for small GQA/MLA configs, including tiny sparse routed-expert setups with optional shared experts,
- a parallel `LuxHeadOnlyDeepSeekV4` bootstrap path plus a first tiny `DeepSeekFullModelTrainerState` path and `DeepSeekv4.jl/scripts/train_deepseek_tiny.jl`,
- an optional DeepSeek Engram branch with dependency-free compressed token lookup support for selected layers,
- shared schedule, batching, next-token loss, Lux-native layer/trainer, checkpoint-layout, and gradient-masking helpers in `TransformerCore.jl`.

That training surface is still intentionally constrained: OpenMythos has the broader current full-model slice, while DeepSeek full-model training is still limited to tiny configs even though it now trains both the primary LM logits path and the current auxiliary MTP heads, with optional Engram layers enabled for small-model experiments. The new `TransformerCore.jl` training helpers are the shared Lux-oriented foundation for refactoring both model families onto one clearer full-model training story.

## Runtime status

The workspace now also has a first reusable long-context runtime seam:

- `TransformerCore.KVCacheEnvelope` to hold a mutable cache dictionary plus `start_pos`,
- `save_kv_cache` and `load_kv_cache` for serialized cache reuse,
- `chunked_prefill` in both `OpenMythos.jl` and `DeepSeekv4.jl`,
- envelope-aware `generate` methods that can resume from a prefetched prompt state,
- growable buffer-backed cache entries that avoid full-tensor concatenation on every append,
- envelope-level capacity hints that let both model families preallocate cache buffers before decode growth begins.

This is still a lightweight reference runtime. It now includes paged cache-buffer internals and preallocated capacity hints, but it does not yet include page-aware attention kernels or production-scale cache management.

## Future work

The training foundation is now in place. The main remaining directions are:

- broader full-model training beyond the current OpenMythos and tiny DeepSeek bootstrap slices,
- paged/preallocated cache internals and longer-context serving work,
- distributed training/runtime behavior,
- deeper performance work,
- broader training/runtime work on top of the new API-documented source surface.
