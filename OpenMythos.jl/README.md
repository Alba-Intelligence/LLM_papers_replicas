# OpenMythos.jl

Julia reimplementation of the Python OpenMythos reference in `../reference/OpenMythos`.

## Current status

The replica now includes:

- the core model stack (`MythosConfig`, attention backends, MoE, recurrent block, `OpenMythos`, generation),
- tokenizer parity through a pragmatic Hugging Face bridge,
- an optional Python-vs-Julia parity harness for selected utilities,
- bootstrap training paths with checkpointing and local/FineWeb smoke data flows,
- chunked prefill plus serializable KV-cache envelopes for cached generation reuse,
- lower-allocation buffer-backed cache growth behind the existing generation API.

The current training surface is intentionally staged: `OpenMythos.jl` now has both the original **Lux-backed head-only** bootstrap path and a first **dense full-model** bootstrap path for tiny GQA configs (`n_experts == 1`, `n_shared_experts == 0`, `n_experts_per_tok == 1`), while the broader sparse/DeepSeek full-model story remains future work.

## Quickstart

### Run the test suite

```bash
OPENMYTHOS_TEST_TOKENIZER_MODEL_ID=gpt2 julia --project=. -q -e 'using Pkg; Pkg.test()'
```

### Tiny forward pass

```julia
using OpenMythos

cfg = bootstrap_training_config(256; seq_len=16, attn_type="gqa")
model = OpenMythos(cfg)
input_ids = reshape(collect(0:15), 1, :)

logits = model(input_ids; n_loops=2)
@show size(logits)  # (1, 16, 256)
```

### Pluto notebook example

There is also a small Pluto notebook at `notebooks/small_example.jl`.

```julia
using Pluto
Pluto.run()
```

Then open `notebooks/small_example.jl`.

### Tokenizer usage

```julia
using OpenMythos

tok = MythosTokenizer("gpt2")
ids = tokenize(tok, "OpenMythos in Julia")
text = detokenize(tok, ids)
@show vocab_size(tok), ids, text
```

### Bootstrap training smoke run

```bash
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=8 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

To exercise the current dense full-model bootstrap slice instead of the default head-only path:

```bash
OPENMYTHOS_TRAIN_MODE=full_model \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=4 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

For an optional FineWeb-Edu-backed smoke run:

```bash
OPENMYTHOS_USE_FINEWEB_EDU=1 \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_FINEWEB_SUBSET=sample-10BT \
OPENMYTHOS_FINEWEB_BATCHES=8 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

### Chunked prefill and cache reuse

```julia
using OpenMythos

cfg = bootstrap_training_config(256; seq_len=16, attn_type="gqa")
model = OpenMythos(cfg)
ids = reshape(collect(0:7), 1, :)

env = chunked_prefill(model, ids; chunk_size=3, n_loops=2)
save_kv_cache(env, "cache/openmythos_prefill.jls")
ids2 = generate(model, ids; max_new_tokens=4, n_loops=2, envelope=load_kv_cache("cache/openmythos_prefill.jls"))
```

## Repository map

- `src/` - Julia package implementation
- `test/` - translated parity and regression tests
- `../docs/wiki/` - shared architecture, usage, and porting notes for the workspace
- `notebooks/small_example.jl` - Pluto notebook smoke example
- `scripts/train_3b_fineweb_edu.jl` - Lux-backed bootstrap training entrypoint
- `../reference/OpenMythos/` - upstream Python reference

## What is still deferred

- broader full-model training beyond the current dense OpenMythos bootstrap slice,
- distributed training/runtime behavior,
- paged or preallocated KV cache internals,
- experimental `moda.py` parity,
- performance-focused optimization work.

## Reading order

1. `../docs/wiki/index.md`
2. `../docs/wiki/usage.md`
3. `../docs/wiki/architecture.md`
4. `../docs/wiki/python-reference-map.md`
5. `../docs/wiki/julia-reimplementation-plan.md`
