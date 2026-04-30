# DeepSeekV4.jl

Architecture-first Julia implementation workspace for DeepSeek V4.

This package currently targets:

- tiny-config forward passes,
- readable CSA/HCA reference attention,
- manifold-constrained hyper-connections,
- DeepSeek-style MoE routing surfaces,
- MTP output heads and generation smoke paths,
- chunked prefill plus serializable KV-cache envelopes for cached generation reuse,
- lower-allocation buffer-backed cache growth behind the existing generation API,
- a Lux-backed head-only bootstrap training surface for tiny configs.

## Quickstart

### Run the test suite

```bash
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### Bootstrap training smoke run

```bash
DEEPSEEK_V4_TRAIN_TOTAL_STEPS=8 \
DEEPSEEK_V4_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_deepseek_tiny.jl
```

This script currently builds local byte-encoded batches for a simple trainable smoke path.

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
