# Manual

## Workspace layout

This repository is a multi-package Julia workspace:

- `TransformerCore.jl/` holds reusable tensor helpers, RoPE, schedules, batching,
  and KV-cache infrastructure.
- `OpenMythos.jl/` holds the recurrent-depth OpenMythos model family.
- `DeepSeekv4.jl/` holds the DeepSeek V4 architecture-first implementation.
- `docs/wiki/` remains the shared narrative design/wiki layer.

## Running tests

```bash
cd TransformerCore.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'

cd ../OpenMythos.jl
OPENMYTHOS_TEST_TOKENIZER_MODEL_ID=gpt2 julia --project=. -q -e 'using Pkg; Pkg.test()'

cd ../DeepSeekv4.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

## Runtime workflow

Both model packages expose the same cache-oriented outer runtime contract:

1. build a `KVCacheEnvelope`,
2. prefill it with `chunked_prefill`,
3. optionally persist it with `save_kv_cache`,
4. resume generation with `generate(...; envelope=...)`.

Example with OpenMythos:

```julia
using OpenMythos

cfg = bootstrap_training_config(256; seq_len=16, attn_type="gqa")
model = OpenMythos(cfg)
ids = reshape(collect(0:7), 1, :)

env = KVCacheEnvelope(; capacity_hint=64)
env = chunked_prefill(model, ids; chunk_size=3, n_loops=2, envelope=env)
continued = generate(model, ids; max_new_tokens=4, n_loops=2, envelope=env)
```

The same `KVCacheEnvelope` / `chunked_prefill` / `generate` workflow also applies
to `DeepSeekV4Model`.

## Training workflow

The current training surface is intentionally bootstrap-sized:

- `OpenMythos.jl` supports both a **head-only Lux path** and a first **dense
  full-model** training slice for small GQA/MLA configs, including tiny sparse
  routed-expert setups with optional shared experts.
- `DeepSeekv4.jl` now exposes both a **head-only Lux path** and a first **tiny
  full-model** bootstrap trainer for the primary LM logits path, the current
  MTP heads, and an optional gated Engram branch.
- `TransformerCore.jl` provides the shared warmup/cosine schedule, token-pair
  batching helpers, checkpoint discovery, and head-loss math.

## Narrative documentation

For architecture discussion, roadmap details, and source/reference mapping, read
the markdown wiki under `docs/wiki/` in the repository root.
