# Usage

This page shows the current user-facing workflow for the Julia replica.

## 1. Construct a tiny model

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

## 2. Use the tokenizer

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

## 3. Open the notebook example

There is a small Pluto notebook in:

```text
notebooks/small_example.jl
```

To open it:

```julia
using Pluto
Pluto.run()
```

Then select `notebooks/small_example.jl`.

The notebook demonstrates:

- project activation,
- tiny-model construction,
- a forward pass,
- short random-weight generation.

## 4. Run the bootstrap training script

The training entrypoint is:

```bash
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
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=8 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

Example FineWeb-Edu-backed smoke run:

```bash
OPENMYTHOS_USE_FINEWEB_EDU=1 \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_FINEWEB_SUBSET=sample-10BT \
OPENMYTHOS_FINEWEB_BATCHES=8 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

## 5. Understand the training scope

The current bootstrap trainer is intentionally limited:

- it is **Lux-backed**,
- it uses `Optimisers.AdamW` and `NNlib.logsoftmax`,
- it currently optimizes the LM head only through `LuxHeadOnlyOpenMythos`,
- the core model internals are still manual Julia blocks,
- full-model autodiff/distributed training is still future work.

## 6. What to read next

1. [Architecture](architecture.md)
2. [Python reference map](python-reference-map.md)
3. [Julia reimplementation plan](julia-reimplementation-plan.md)
