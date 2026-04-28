# AGENTS.md

## Mission

This repository is for a Julia reimplementation of the Python project stored in `reference/OpenMythos`.

The main job for coding agents is to turn the Python reference into a Julia package and documentation set without losing the architectural core:

- Prelude -> Recurrent Block -> Coda,
- ACT halting,
- LTI-stable input injection,
- loop-index embedding,
- depth-wise LoRA,
- MoE recurrent FFN,
- switchable GQA / MLA attention.

## Current state

- Root repo: environment scaffolding, Julia package scaffold, and porting docs.
- Python reference: `reference/OpenMythos`.
- Julia package: `Project.toml`, `src/`, `test/`, and `Manifest.toml` now exist.
- Wiki: `docs/wiki/`.

The current Julia slice covers the core numerical primitives and the main model stack:

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
- `loop_index_embedding`
- `LoRAAdapter`
- `LTIInjection`
- `ACTHalting`

Training/data integration is still future work.

## Source priority

Use sources in this order:

1. `reference/OpenMythos/open_mythos/main.py`
2. `reference/OpenMythos/docs/open_mythos.md`
3. `reference/OpenMythos/tests/test_main.py`
4. `reference/OpenMythos/open_mythos/variants.py`
5. `reference/OpenMythos/tests/test_tokenizer.py`
6. `reference/OpenMythos/training/3b_fine_web_edu.py`
7. `reference/OpenMythos/README.md`

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
- benchmark scripts under `reference/OpenMythos/tests/`

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

## Expected repository shape

Agents should grow the repo toward:

```text
Project.toml
src/
  OpenMythos.jl
  config.jl
  norms.jl
  rope.jl
  blocks.jl
  recurrent.jl
  moe.jl
  model.jl
  variants.jl
  tokenizer.jl
  attention/
    gqa.jl
    mla.jl
test/
  runtests.jl
docs/wiki/
scripts/
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

## Current reference commands

These commands exist only in the Python reference subtree:

```bash
cd reference/OpenMythos
python -m pytest tests/test_main.py -q
python -m pytest tests/test_tokenizer.py -q
python training/3b_fine_web_edu.py
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

If installed, the Python reference also advertises `ruff` and `black` via `pyproject.toml`.

Do not present those as root-level Julia commands.

## Near-term execution order

1. Add training/data integration.
2. Bring in Lux-centered training/runtime abstractions as the model stack matures.
3. Polish public docs and package APIs.
