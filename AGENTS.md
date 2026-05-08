# AGENTS.md

## Mission

This repository is for a Julia reimplementation of the Python project stored in `reference/private/OpenMythos` for `OpenMythos.jl` work.

The main job for coding agents is to turn the Python reference into a Julia package and documentation set without losing the architectural core:

- Prelude -> Recurrent Block -> Coda,
- ACT halting,
- LTI-stable input injection,
- loop-index embedding,
- depth-wise LoRA,
- MoE recurrent FFN,
- switchable GQA / MLA attention.

## Current state

- Root repo: environment scaffolding, a multi-package Julia workspace, and porting docs.
- Python reference for `OpenMythos.jl`: `reference/private/OpenMythos`.
- Julia packages:
  - `OpenMythos.jl/` for the recurrent OpenMythos package,
  - `DeepSeekv4.jl/` for the DeepSeek V4 package,
  - `TransformerCore.jl/` for shared reusable primitives,
  - `TextDataCore.jl/` for shared tokenizer and local text-data helpers.
- Wiki: `docs/wiki/`.

The current Julia slice covers the core numerical primitives, the main model stack, tokenizer parity, and a bootstrap training/data path plus a first shared Lux-native training foundation in `TransformerCore.jl`:

- `MythosConfig`
- `RMSNorm`
- RoPE precomputation and application
- `GQAttention`
- `MLAttention`
- `Expert`
- `MoEFFN`
- `TransformerBlock`
- `RecurrentBlock`
- `OpenMythos`
- `MythosTokenizer`
- `WarmupCosineSchedule`
- token chunking / next-token batching helpers
- head-only bootstrap training loop
- generic next-token Lux trainer / gradient-mask helpers in `TransformerCore.jl`
- family/mode-aware checkpoint save / load helpers
- `loop_index_embedding`
- `LoRAAdapter`
- `LTIInjection`
- `ACTHalting`
- Lux-native OpenMythos mirrors for attention, experts / MoE, transformer blocks, recurrent update primitives, the tied-embedding model shell, and a first shared-trainer-backed Lux full-model state with shared save/load wrappers

The current OpenMythos training path is intentionally bootstrap-sized: it wires tokenizer/data batching, checkpointing, a Julia `OpenMythos.jl/scripts/train_3b_fineweb_edu.jl` entrypoint, a Julia-native FineWeb dataset-viewer rows path for remote smoke loading, a Julia-native local-parquet FineWeb path, a broader small-config full-model path, a first shared Lux-native trainer/checkpoint foundation in `TransformerCore.jl`, Lux-native OpenMythos attention/FFN/block/recurrent/model mirrors, a first `LuxFullModelTrainerState` plus shared save/load wrappers built on the shared trainer surface, and a shared `TextDataCore.jl` BytePairEncoding/tiktoken-backed tokenizer + local text-data path for the currently supported GPT-style families. The Python FineWeb bridge has now been removed from the package path. The script default now points at the Lux-native full-model path while legacy modes keep separate checkpoint directories. `DeepSeekv4.jl` now also reuses the shared tokenizer/data path and shared family/mode-aware checkpoint layout while keeping its own MTP-aware full-model loss/update logic.

## Source priority

Use sources in this order:

1. `reference/private/OpenMythos/open_mythos/main.py`
2. `reference/private/OpenMythos/docs/open_mythos.md`
3. `reference/private/OpenMythos/tests/test_main.py`
4. `reference/private/OpenMythos/open_mythos/variants.py`
5. `reference/private/OpenMythos/tests/test_tokenizer.py`
6. `reference/private/OpenMythos/training/3b_fine_web_edu.py`
7. `reference/private/OpenMythos/README.md`

`README.md` is useful, but it mixes implementation facts with broader theory and speculation. Prefer code and tests when they disagree.

## Scope rules

### Primary port target

- `open_mythos/main.py`
- `open_mythos/tokenizer.py`
- `open_mythos/variants.py`
- `tests/test_main.py`
- `tests/test_tokenizer.py`

### Secondary target

- `training/3b_fine_web_edu.py`
- `docs/datasets.md`

### Defer unless explicitly requested

- `open_mythos/moda.py`
- `examples/moda_example.py`
- benchmark scripts under `reference/private/OpenMythos/tests/`

`moda.py` is a separate experimental branch and should not distort the first Julia milestone.

## Julia stack

Optimize the implementation plan around:

- `Lux.jl`
- `NNlib.jl`
- `Zygote.jl`
- `Optimisers.jl`
- `MLUtils.jl`
- `CUDA.jl`

The current package slice is stdlib-first so it can build in restricted environments. Bring in the broader Julia ML stack as the model blocks land.

Use `PythonCall.jl` only when it meaningfully reduces bootstrap risk, especially for tokenizer or dataset interop.

The core model implementation is still manual and parity-oriented, but the training surface now uses `Lux.jl`, `NNlib.jl`, and `Optimisers.jl`.

## Expected repository shape

Agents should grow the repo toward:

```text
OpenMythos.jl/
  Project.toml
  src/
  test/
  scripts/
  notebooks/
DeepSeekv4.jl/
  Project.toml
  src/
  test/
TransformerCore.jl/
  Project.toml
  src/
  test/
TextDataCore.jl/
  Project.toml
  src/
  test/
docs/wiki/
```

## Working conventions

### Port invariants first

Translate the Python invariant tests before attempting large-scale training code. The best early targets are:

- RMSNorm behavior,
- RoPE correctness,
- KV-cache logic,
- ACT weighting,
- LoRA depth handling,
- recurrent block shape and state-update behavior.

### Keep configs tiny during bootstrap

Use small configs similar to the Python tests for most early work. Do not begin with 1B+ presets as the development baseline.

### Separate parity from optimization

First match behavior with clear, readable Julia code. Only then optimize kernels, GPU paths, or cache layouts.

### Keep bootstrap training honest

The current Julia training path is a Lux-backed bootstrap bridge, not full PyTorch parity:

- it covers data/tokenizer integration, batching, scheduling, checkpointing, and resumable smoke training,
- it now also has a first shared `TransformerCore.jl` Lux-native trainer/checkpoint foundation intended to replace duplicated package-local training code,
- the current `OpenMythos.jl` and `DeepSeekv4.jl` model internals are still mostly manual Julia blocks under that evolving Lux training surface.

### Preserve architecture names

Use names close to the Python reference where possible:

- `MythosConfig`
- `RMSNorm`
- `GQAttention`
- `MLAttention`
- `MoEFFN`
- `RecurrentBlock`
- `OpenMythos`

That keeps the wiki, tests, and source mapping aligned.

## Documentation expectations

Before or during major implementation work, keep `docs/wiki/` current:

- `index.md` explains the big picture,
- `architecture.md` explains the model,
- `python-reference-map.md` maps source files to Julia work,
- `references.md` centralizes citations,
- `julia-reimplementation-plan.md` records the module plan.

When code changes invalidate those pages, update them.

## Completion workflow

When a task materially changes the package, architecture notes, workflow, or implementation status:

1. Update the relevant `docs/wiki/` pages before considering the task complete.
2. Create a git commit after the task is complete.
3. Prefer logical commit boundaries that preserve a readable port history instead of one large catch-all commit.

## Current reference commands

These commands exist only in the Python reference subtree:

```bash
cd reference/private/OpenMythos
python -m pytest tests/test_main.py -q
python -m pytest tests/test_tokenizer.py -q
python training/3b_fine_web_edu.py
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

If installed, the Python reference also advertises `ruff` and `black` via `pyproject.toml`.

Do not present those as root-level Julia commands.

## Near-term execution order

1. Extend training beyond the current head-only Lux layer.
2. Explore distributed/runtime behavior once the single-process story grows.
3. Expand docs again when those surfaces land.
