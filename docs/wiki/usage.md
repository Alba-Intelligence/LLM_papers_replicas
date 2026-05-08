# Usage

This page shows the current user-facing workflow for the Julia workspace.

## Package locations

- Run OpenMythos package commands from `OpenMythos.jl/`.
- Run DeepSeek V4 package commands from `DeepSeekv4.jl/`.
- Shared low-level primitives plus the emerging Lux-native training/checkpoint foundation live in `TransformerCore.jl/`.
- Use `docs/` for the generated `Documenter.jl` site and `docs/wiki/` for the longer-form narrative docs.

## 1. Run package tests

```bash
cd TransformerCore.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'

cd ../OpenMythos.jl
OPENMYTHOS_TEST_TOKENIZER_MODEL_ID=gpt2 julia --project=. -q -e 'using Pkg; Pkg.test()'

cd ../DeepSeekv4.jl
julia --project=. -q -e 'using Pkg; Pkg.test()'
```

## 2. Build the API documentation

```bash
julia --project=docs -q docs/make.jl
```

This builds the shared `Documenter.jl` site from:

- source docstrings in all three Julia packages,
- manual overview pages under `docs/src/`,
- the shared docs environment in `docs/Project.toml`.

## 3. Construct a tiny OpenMythos model

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

## 4. Construct a tiny DeepSeek V4 model

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

To enable the optional Engram branch on a tiny reference config:

```julia
cfg = deepseek_v4_tiny_engram()
model = DeepSeekV4Model(cfg)
```

## 5. Use the tokenizer

The tokenizer is currently still package-specific and in transition: the current OpenMythos surface remains parity-oriented, while the shared training foundation has now moved into `TransformerCore.jl` so the next package refactor can replace Python bridges with Julia-native paths.

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

## 6. Reuse a prefetched KV cache

Both model packages now expose the same lightweight runtime workflow:

```julia
using OpenMythos

cfg = bootstrap_training_config(256; seq_len=16, attn_type="gqa")
model = OpenMythos(cfg)
ids = reshape(collect(0:7), 1, :)

env = chunked_prefill(model, ids; chunk_size=3, n_loops=2)
save_kv_cache(env, "cache/openmythos_prefill.jls")

loaded = load_kv_cache("cache/openmythos_prefill.jls")
continued = generate(model, ids; max_new_tokens=4, n_loops=2, envelope=loaded)
```

The same `chunked_prefill`, `save_kv_cache`, `load_kv_cache`, and `generate(...; envelope=...)` pattern also works in `DeepSeekV4.jl`. If you want the cache to reserve space ahead of decode growth, construct the envelope with a capacity hint:

```julia
env = KVCacheEnvelope(; capacity_hint=256)
env = chunked_prefill(model, ids; chunk_size=8, envelope=env)
continued = generate(model, ids; max_new_tokens=32, envelope=env)
```

This is a reference runtime seam, not a production serving stack: the envelope still owns a Julia dictionary, but the per-layer cache payloads now use growable buffer-backed entries and can reserve capacity ahead of time rather than growing from minimal allocations on every decode step.

## 7. Open the notebook example

There is a small Pluto notebook in:

```text
notebooks/openmythos/small_example.jl
```

To open it:

```julia
using Pluto
Pluto.run()
```

Then select `notebooks/openmythos/small_example.jl`.

The notebook demonstrates:

- project activation,
- tiny-model construction,
- a forward pass,
- short random-weight generation.

## 8. Run the bootstrap training scripts

The OpenMythos training entrypoint is:

```bash
cd OpenMythos.jl
julia --project=. scripts/train_3b_fineweb_edu.jl
```

Useful environment variables:

| Variable                              | Meaning                                              | Default                                |
| ------------------------------------- | ---------------------------------------------------- | -------------------------------------- |
| `OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID` | tokenizer model ID                                                           | `openai/gpt-oss-20b` unless overridden |
| `OPENMYTHOS_TRAIN_MODE`               | training mode (`full_model`, `full_model_lux`, `full_model_legacy`, `head_only`) | `full_model`                           |
| `OPENMYTHOS_TRAIN_TOTAL_STEPS`        | total bootstrap steps                                                        | `8`                                    |
| `OPENMYTHOS_TRAIN_SEQ_LEN`            | sequence length                                                              | `32`                                   |
| `OPENMYTHOS_TRAIN_BATCH_SIZE`         | batch size                                                                   | `2`                                    |
| `OPENMYTHOS_TRAIN_ATTN_TYPE`          | attention backend (`gqa` or `mla`)                                           | `gqa`                                  |
| `OPENMYTHOS_TRAIN_N_EXPERTS`          | routed experts in full-model mode                                            | `1`                                    |
| `OPENMYTHOS_TRAIN_SHARED_EXPERTS`     | shared experts in full-model mode                                            | `0`                                    |
| `OPENMYTHOS_TRAIN_EXPERTS_PER_TOKEN`  | routed experts selected per token in full-model mode                         | `1`                                    |
| `OPENMYTHOS_TRAIN_CKPT_DIR`           | checkpoint root directory (`openmythos/full_model_lux` for Lux mode; `<mode>/` subdirs for legacy modes) | `checkpoints`                          |
| `OPENMYTHOS_USE_FINEWEB_EDU`          | use FineWeb-Edu batch bridge                                                 | `0`                                    |
| `OPENMYTHOS_FINEWEB_SUBSET`           | FineWeb-Edu subset                                                           | `sample-10BT`                          |
| `OPENMYTHOS_FINEWEB_BATCHES`          | number of FineWeb batches to fetch                                           | `max(total_steps, 1)`                  |

Example local-text smoke run using the default Lux-native full-model path:

```bash
cd OpenMythos.jl
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=8 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

Mode aliases:

- `OPENMYTHOS_TRAIN_MODE=full_model` -> Lux-native full-model path (default)
- `OPENMYTHOS_TRAIN_MODE=full_model_lux` -> explicit Lux-native full-model path
- `OPENMYTHOS_TRAIN_MODE=full_model_legacy` -> legacy mutable full-model trainer
- `OPENMYTHOS_TRAIN_MODE=head_only` -> legacy head-only trainer

The script now keeps Lux full-model checkpoints under the shared family/mode-aware layout and isolates legacy modes into separate subdirectories under `OPENMYTHOS_TRAIN_CKPT_DIR` to avoid mixed checkpoint namespaces.

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

| Variable                        | Meaning                                     | Default              |
| ------------------------------- | ------------------------------------------- | -------------------- |
| `DEEPSEEK_V4_TRAIN_VOCAB_SIZE`  | tiny bootstrap vocab size                   | `512`                |
| `DEEPSEEK_V4_TRAIN_MODE`        | training mode (`head_only` or `full_model`) | `head_only`          |
| `DEEPSEEK_V4_TRAIN_USE_ENGRAM`  | enable the gated Engram branch              | `0`                  |
| `DEEPSEEK_V4_TRAIN_TOTAL_STEPS` | total bootstrap steps                       | `8`                  |
| `DEEPSEEK_V4_TRAIN_SEQ_LEN`     | sequence length                             | `32`                 |
| `DEEPSEEK_V4_TRAIN_BATCH_SIZE`  | batch size                                  | `2`                  |
| `DEEPSEEK_V4_TRAIN_CKPT_DIR`    | checkpoint directory                        | `checkpoints/<mode>` |
| `DEEPSEEK_V4_TRAIN_TEXT`        | local training text override                | empty                |
| `DEEPSEEK_V4_TRAIN_TEXT_FILE`   | path to local training text                 | empty                |

Example local DeepSeek smoke run:

```bash
cd DeepSeekv4.jl
DEEPSEEK_V4_TRAIN_MODE=full_model \
DEEPSEEK_V4_TRAIN_USE_ENGRAM=1 \
DEEPSEEK_V4_TRAIN_TOTAL_STEPS=8 \
DEEPSEEK_V4_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_deepseek_tiny.jl
```

## 9. Understand the current training and runtime scope

The current bootstrap trainer is intentionally limited:

- shared schedule, batching, next-token loss, Lux-native trainer, gradient-masking, and family/mode-aware checkpoint helpers now live in `TransformerCore.jl`,
- the original bootstrap path is **Lux-backed**,
- it uses `Optimisers.AdamW`,
- `OpenMythos.jl` now defaults its script surface to a Lux-native dense full-model bootstrap mode via `OPENMYTHOS_TRAIN_MODE=full_model`,
- the package now also exposes a first Lux-native `LuxFullModelTrainerState` built on `TransformerCore.NextTokenTrainerState` plus shared family/mode-aware checkpoint save/load helpers,
- those full-model paths currently support small GQA/MLA configs, including tiny sparse routed-expert setups with optional shared experts, while `full_model_legacy` and `head_only` remain compatibility modes,
- `DeepSeekv4.jl` now has both a `LuxHeadOnlyDeepSeekV4` head-only trainer and a first tiny `DeepSeekFullModelTrainerState` bootstrap path,
- the current DeepSeek full-model loss now trains both the main LM logits path and the current auxiliary MTP heads on tiny configs, with an optional gated Engram branch,
- the core model internals are still mostly manual Julia blocks, though `OpenMythos.jl` now also exposes Lux-native mirrors for attention, experts / MoE, transformer blocks, recurrent update primitives, and a tied-embedding `LuxOpenMythos` shell,
- broader sparse/full-model autodiff and distributed training are still future work.

The current runtime seam is also intentionally lightweight:

- `KVCacheEnvelope` provides a shared outer cache contract,
- both packages support `chunked_prefill`,
- cache payloads are still family-specific runtime entries under a `Dict{String, Any}` envelope, now backed by growable append buffers with optional preallocated capacity hints,
- true paged attention and production cache allocators are still future work.

## 9. What to read next

1. [Architecture](architecture.md)
2. [Python reference map](python-reference-map.md)
3. [DeepSeek V4 architecture](deepseek-v4-architecture.md)
4. [DeepSeek V4 reference map](deepseek-v4-reference-map.md)
5. [Julia reimplementation plan](julia-reimplementation-plan.md)
