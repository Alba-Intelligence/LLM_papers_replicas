# OpenMythos.jl

Julia reimplementation of the Python OpenMythos reference in `reference/OpenMythos`.

## Current status

The replica now includes:

- the core model stack (`MythosConfig`, attention backends, MoE, recurrent block, `OpenMythos`, generation),
- tokenizer parity through a pragmatic Hugging Face bridge,
- an optional Python-vs-Julia parity harness for selected utilities,
- a Lux-backed bootstrap training path with checkpointing and local/FineWeb smoke data flows.

The current training surface is intentionally **head-only**: the core model implementation remains manual and parity-oriented, while the bootstrap trainer uses `Lux.jl`, `NNlib.jl`, and `Optimisers.jl`.

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

For an optional FineWeb-Edu-backed smoke run:

```bash
OPENMYTHOS_USE_FINEWEB_EDU=1 \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_FINEWEB_SUBSET=sample-10BT \
OPENMYTHOS_FINEWEB_BATCHES=8 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

## Repository map

- `src/` - Julia package implementation
- `test/` - translated parity and regression tests
- `docs/wiki/` - architecture, usage, and porting notes
- `scripts/train_3b_fineweb_edu.jl` - Lux-backed bootstrap training entrypoint
- `reference/OpenMythos/` - upstream Python reference (ignored by git in this replica repo)

## What is still deferred

- full-model gradient-based training beyond the head-only bootstrap layer,
- distributed training/runtime behavior,
- experimental `moda.py` parity,
- performance-focused optimization work.

## Reading order

1. `docs/wiki/index.md`
2. `docs/wiki/usage.md`
3. `docs/wiki/architecture.md`
4. `docs/wiki/python-reference-map.md`
5. `docs/wiki/julia-reimplementation-plan.md`
