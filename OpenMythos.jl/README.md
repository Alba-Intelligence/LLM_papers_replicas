# OpenMythos.jl

Julia reimplementation of the vendored Python OpenMythos reference in `../reference/private/OpenMythos`.

## Current status

The replica now includes:

- the core model stack (`MythosConfig`, attention backends, MoE, recurrent block, `OpenMythos`, generation),
- the first Lux-native mirrors for attention, experts / MoE, transformer blocks, recurrent update primitives, and the tied-embedding model shell,
- tokenizer parity through a Julia-native BytePairEncoding / tiktoken-backed path,
- an optional Python-vs-Julia parity harness for selected utilities,
- bootstrap training paths with checkpointing and local/FineWeb smoke data flows,
- chunked prefill plus serializable KV-cache envelopes for cached generation reuse,
- lower-allocation buffer-backed cache growth plus envelope-level preallocation hints behind the existing generation API,
- true paged cache-buffer internals beneath the shared envelope API,
- a broadened full-model bootstrap slice that now supports both GQA and MLA attention plus small sparse routed-expert configs with optional shared experts,
- a first Lux-native `LuxFullModelTrainerState` built on the shared `TransformerCore.NextTokenTrainerState` foundation.

The current training surface is intentionally staged: `OpenMythos.jl` now has both the original **Lux-backed head-only** bootstrap path and a broader **full-model** bootstrap path for small GQA/MLA configs, including tiny sparse routed-expert setups with optional shared experts. The package also now includes Lux-native mirrors (`LuxGQAttention`, `LuxMLAttention`, `LuxExpert`, `LuxMoEFFN`, `LuxTransformerBlock`, `LuxRecurrentBlock`, `LuxLoRAAdapter`, `LuxLTIInjection`, `LuxACTHalting`, `LuxOpenMythos`) plus a first `LuxFullModelTrainerState` that reuses the current parity stack through the shared `TransformerCore.jl` trainer foundation and shared family/mode-aware checkpoints. The tokenizer path is now Julia-native for supported GPT/tiktoken families, and the FineWeb path now also has a Julia-native local parquet option; the remaining Python bridge is only the optional remote FineWeb streaming path. The training script now treats Lux-native full-model training as the default path; the legacy mutable full-model trainer remains available as an explicit compatibility mode.

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

The default script path is now the **Lux-native full-model** bootstrap trainer.

```bash
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=8 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

The mode aliases are:

- `OPENMYTHOS_TRAIN_MODE=full_model` -> Lux-native full-model path (default)
- `OPENMYTHOS_TRAIN_MODE=full_model_lux` -> explicit Lux-native full-model path
- `OPENMYTHOS_TRAIN_MODE=full_model_legacy` -> legacy mutable full-model trainer
- `OPENMYTHOS_TRAIN_MODE=head_only` -> legacy head-only trainer

The script now stores Lux full-model checkpoints under the shared family/mode-aware layout and isolates legacy modes into their own subdirectories under `OPENMYTHOS_TRAIN_CKPT_DIR` so checkpoint namespaces do not collide.

To exercise the current Lux-native dense full-model bootstrap slice explicitly:

```bash
OPENMYTHOS_TRAIN_MODE=full_model_lux \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=4 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

To exercise the MLA-backed Lux-native full-model slice with one shared expert:

```bash
OPENMYTHOS_TRAIN_MODE=full_model_lux \
OPENMYTHOS_TRAIN_ATTN_TYPE=mla \
OPENMYTHOS_TRAIN_SHARED_EXPERTS=1 \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=4 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

To exercise a small sparse Lux-native full-model slice:

```bash
OPENMYTHOS_TRAIN_MODE=full_model_lux \
OPENMYTHOS_TRAIN_ATTN_TYPE=gqa \
OPENMYTHOS_TRAIN_N_EXPERTS=4 \
OPENMYTHOS_TRAIN_SHARED_EXPERTS=1 \
OPENMYTHOS_TRAIN_EXPERTS_PER_TOKEN=2 \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=4 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

To exercise the legacy mutable full-model trainer explicitly:

```bash
OPENMYTHOS_TRAIN_MODE=full_model_legacy \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_TRAIN_TOTAL_STEPS=4 \
OPENMYTHOS_TRAIN_SEQ_LEN=32 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

For an optional FineWeb-Edu-backed smoke run through the existing Python streaming bridge:

```bash
OPENMYTHOS_USE_FINEWEB_EDU=1 \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
OPENMYTHOS_FINEWEB_SUBSET=sample-10BT \
OPENMYTHOS_FINEWEB_BATCHES=8 \
julia --project=. scripts/train_3b_fineweb_edu.jl
```

For a Julia-native FineWeb-style smoke run from local parquet shard(s):

```bash
OPENMYTHOS_USE_FINEWEB_EDU=1 \
OPENMYTHOS_FINEWEB_PARQUET_PATH=/path/to/fineweb-sample.parquet \
OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID=gpt2 \
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
- `../reference/private/OpenMythos/` - vendored Python reference

## What is still deferred

- DeepSeek full-model training beyond the current OpenMythos bootstrap slice,
- distributed training/runtime behavior,
- page-aware attention kernels and deeper cache lifecycle/runtime work,
- experimental `moda.py` parity,
- performance-focused optimization work.

## Reading order

1. `../docs/wiki/index.md`
2. `../docs/wiki/usage.md`
3. `../docs/wiki/architecture.md`
4. `../docs/wiki/python-reference-map.md`
5. `../docs/wiki/julia-reimplementation-plan.md`
