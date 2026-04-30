# Usage

This page shows the current user-facing workflow for the Julia workspace.

## Package locations

- Run OpenMythos package commands from `OpenMythos.jl/`.
- Run DeepSeek V4 package commands from `DeepSeekv4.jl/`.
- Shared low-level primitives and generic training helpers live in `TransformerCore.jl/`.
- Keep using the shared wiki from the repository root.

## 1. Run package tests

```bash
cd TransformerCore.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'

cd ../OpenMythos.jl
OPENMYTHOS_TEST_TOKENIZER_MODEL_ID=gpt2 julia --project=. -q -e 'using Pkg; Pkg.test()'

cd ../DeepSeekv4.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

## 2. Construct a tiny OpenMythos model

For smoke runs and quick experimentation, prefer the bootstrap-sized config helpers over the large reference presets.

```julia
using OpenMythos

cfg = bootstrap_training_config(256; seq_len=16, attn_type="gqa")
model = OpenMythos(cfg)
input_ids = reshape(collect(0:15), 1, :)

logits = model(input_ids; n_loops=2)
size(logits)
```

Expected shape:

```text
(1, 16, 256)
```

## 3. Construct a tiny DeepSeek V4 model

```julia
using DeepSeekV4

cfg = deepseek_v4_tiny()
model = DeepSeekV4Model(cfg)
input_ids = reshape(collect(0:7), 1, :)

logits = model(input_ids)
size(logits)
```

Expected shape:

```text
(1, 8, cfg.vocab_size)
```

## 4. Use the tokenizer

The tokenizer currently shells out through Hugging Face tooling for parity and practicality.

```julia
using OpenMythos

tok = MythosTokenizer("gpt2")
ids = tokenize(tok, "OpenMythos in Julia")
text = detokenize(tok, ids)
```

Useful entry points:

- `MythosTokenizer(model_id)`
- `tokenize(tok, text)`
- `detokenize(tok, ids)`
- `vocab_size(tok)`

## 5. Open the notebook example

There is a small Pluto notebook in:

```text
OpenMythos.jl/notebooks/small_example.jl
```

To open it:

```julia
using Pluto
Pluto.run()
```

Then select `OpenMythos.jl/notebooks/small_example.jl`.

The notebook demonstrates:

- project activation,
- tiny-model construction,
- a forward pass,
- short random-weight generation.

## 6. Run the bootstrap training scripts

The OpenMythos training entrypoint is:

```bash
cd OpenMythos.jl
julia --project=. scripts/train_3b_fineweb_edu.jl
```

Useful environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID` | tokenizer model ID | `openai/gpt-oss-20b` unless overridden |
| `OPENMYTHOS_TRAIN_TOTAL_STEPS` | total bootstrap steps | `8` |
| `OPENMYTHOS_TRAIN_SEQ_LEN` | sequence length | `32` |
| `OPENMYTHOS_TRAIN_BATCH_SIZE` | batch size | `2` |
| `OPENMYTHOS_TRAIN_CKPT_DIR` | checkpoint directory | `checkpoints` |
| `OPENMYTHOS_USE_FINEWEB_EDU` | use FineWeb-Edu batch bridge | `0` |
| `OPENMYTHOS_FINEWEB_SUBSET` | FineWeb-Edu subset | `sample-10BT` |
| `OPENMYTHOS_FINEWEB_BATCHES` | number of FineWeb batches to fetch | `max(total_steps, 1)` |

Example local-text smoke run:

```bash
cd OpenMythos.jl
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=8 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

Example FineWeb-Edu-backed smoke run:

```bash
cd OpenMythos.jl
OPENMYTHOS_USE_FINEWEB_EDU=1 \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_FINEWEB_SUBSET=sample-10BT \
OPENMYTHOS_FINEWEB_BATCHES=8 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

The DeepSeek V4 training entrypoint is:

```bash
cd DeepSeekv4.jl
julia --project=. scripts/train_deepseek_tiny.jl
```

Useful environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `DEEPSEEK_V4_TRAIN_VOCAB_SIZE` | tiny bootstrap vocab size | `512` |
| `DEEPSEEK_V4_TRAIN_TOTAL_STEPS` | total bootstrap steps | `8` |
| `DEEPSEEK_V4_TRAIN_SEQ_LEN` | sequence length | `32` |
| `DEEPSEEK_V4_TRAIN_BATCH_SIZE` | batch size | `2` |
| `DEEPSEEK_V4_TRAIN_CKPT_DIR` | checkpoint directory | `checkpoints` |
| `DEEPSEEK_V4_TRAIN_TEXT` | local training text override | empty |
| `DEEPSEEK_V4_TRAIN_TEXT_FILE` | path to local training text | empty |

Example local DeepSeek smoke run:

```bash
cd DeepSeekv4.jl
DEEPSEEK_V4_TRAIN_TOTAL_STEPS=8 \
DEEPSEEK_V4_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_deepseek_tiny.jl
```

## 7. Understand the training scope

The current bootstrap trainer is intentionally limited:

- shared schedule, batching, checkpoint scanning, and head-loss math now live in `TransformerCore.jl`,
- it is **Lux-backed**,
- it uses `Optimisers.AdamW`,
- it currently optimizes the LM head only through `LuxHeadOnlyOpenMythos`,
- `DeepSeekv4.jl` now has a parallel `LuxHeadOnlyDeepSeekV4` head-only trainer and smoke script,
- the core model internals are still manual Julia blocks,
- full-model autodiff/distributed training is still future work.

## 8. What to read next

1. [Architecture](architecture.md)
2. [Python reference map](python-reference-map.md)
3. [DeepSeek V4 architecture](deepseek-v4-architecture.md)
4. [DeepSeek V4 reference map](deepseek-v4-reference-map.md)
5. [Julia reimplementation plan](julia-reimplementation-plan.md)
