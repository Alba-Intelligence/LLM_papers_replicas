# DeepSeekV4.jl

Architecture-first Julia implementation workspace for DeepSeek V4.

This package currently targets:

- tiny-config forward passes,
- readable CSA/HCA reference attention,
- manifold-constrained hyper-connections,
- DeepSeek-style MoE routing surfaces,
- an optional gated Engram conditional-memory branch with token-id hashing and compressed token lookup support,
- MTP output heads and generation smoke paths,
- chunked prefill plus serializable KV-cache envelopes for cached generation reuse,
- lower-allocation buffer-backed cache growth behind the existing generation API,
- a Lux-backed head-only bootstrap training surface for tiny configs,
- a first tiny full-model bootstrap trainer for the main LM logits path, auxiliary MTP heads, and optional Engram-enabled stacks.

## Quickstart

### Run the test suite

```bash
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### Bootstrap training smoke run

```bash
DEEPSEEK_V4_TRAIN_MODE=full_model \
DEEPSEEK_V4_TRAIN_USE_ENGRAM=1 \
DEEPSEEK_V4_TRAIN_TOTAL_STEPS=8 \
DEEPSEEK_V4_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_deepseek_tiny.jl
```

This script currently builds local byte-encoded batches for either a head-only or
tiny full-model smoke path. The full-model mode optimizes both the primary LM
head path and the current auxiliary MTP heads on tiny configs, and it can also
enable the gated Engram branch with `DEEPSEEK_V4_TRAIN_USE_ENGRAM=1`.

### Chunked prefill and cache reuse

```julia
using DeepSeekV4

cfg = deepseek_v4_tiny()
model = DeepSeekV4Model(cfg)
ids = reshape(collect(0:7), 1, :)

env = chunked_prefill(model, ids; chunk_size=3)
save_kv_cache(env, "cache/deepseek_prefill.jls")
ids2 = generate(model, ids; max_new_tokens=4, envelope=load_kv_cache("cache/deepseek_prefill.jls"))
```

The shared wiki for this repository remains in `../docs/wiki/`.
