# Julia Reimplementation Plan

## Goal

Rebuild the main OpenMythos model family in Julia while preserving the behavioral core of the Python reference and using Julia-native libraries where possible.

## Current implementation status

The repository now has a working Julia package scaffold plus a parity-tested core model stack:

- `MythosConfig`
- `RMSNorm`
- `precompute_rope_freqs`
- `apply_rope`
- `GQAttention`
- `MLAttention`
- `Expert`
- `MoEFFN`
- `TransformerBlock`
- `RecurrentBlock`
- `OpenMythos`
- `loop_index_embedding`
- `LoRAAdapter`
- `LTIInjection`
- `ACTHalting`

The remaining work is training/data integration and the later Lux-centered training/runtime refactor.

## Chosen stack

| Concern | Julia library |
| --- | --- |
| Model definition | `Lux.jl` |
| Tensor ops and NN primitives | `NNlib.jl` |
| Autodiff | `Zygote.jl` |
| Optimizers | `Optimisers.jl` |
| Data loading | `MLUtils.jl` |
| GPU execution | `CUDA.jl` |
| Documentation | `Documenter.jl` later, `docs/wiki` now |
| Serialization | `JLD2.jl` or another Julia-native checkpoint format |
| Temporary Python interop where needed | `PythonCall.jl` |

## Porting principles

1. Preserve invariants before chasing training scale.
2. Keep the first milestone centered on `open_mythos/main.py`, not `moda.py`.
3. Start with small CPU-friendly configs mirroring the Python tests.
4. Separate parity work from optimization work.
5. Use Python interop only where it reduces risk during the bootstrap phase.
6. After completing a task, update the affected wiki pages before making the follow-up git commit.

## Proposed target layout

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
  test_norms.jl
  test_rope.jl
  test_attention_gqa.jl
  test_attention_mla.jl
  test_moe.jl
  test_recurrent.jl
  test_model.jl
  test_tokenizer.jl
scripts/
  train_3b_fineweb_edu.jl
docs/
  wiki/
```

## Phase plan

### Phase 1: package scaffold

- Create `Project.toml`.
- Create `src/OpenMythos.jl` and module includes.
- Add `test/runtests.jl`.
- Set up a minimal `docs/` structure that can grow into `Documenter.jl`.

Status: done.

### Phase 2: numerical primitives

Port the pieces with clear local invariants:

- `MythosConfig`
- `RMSNorm`
- `precompute_rope_freqs`
- `apply_rope`
- `loop_index_embedding`
- `LTIInjection`
- `ACTHalting`
- `LoRAAdapter`

These should be validated against the Python tests first because they are the most portable and the least dependent on framework-level details.

Status: done for the initial primitive set listed above.

### Phase 3: attention and FFN blocks

- Implement `GQAttention` first as the simpler attention path.
- Implement `MLAttention` next because it is the default path in the reference.
- Implement dense `Expert`.
- Implement `MoEFFN`.
- Implement `TransformerBlock`.

At this stage, focus on correctness and shape parity rather than full performance parity with PyTorch or Flash Attention.

Status: done for the current GQA/MLA + FFN stack.

### Phase 4: recurrent block and full model

- Implement `RecurrentBlock`.
- Implement `OpenMythos`.
- Implement generation and KV-cache handling.
- Port variant presets from `variants.py`.

The biggest behavioral checkpoints here are:

- ACT weighting,
- loop-index handling,
- cache key separation across recurrent depth,
- safe behavior when `n_loops` exceeds trained loop embeddings.

Status: done for forward/generate parity on small test configurations.

### Phase 5: tokenizer and data access

- Start with a pragmatic tokenizer bridge if needed:
  - temporary `PythonCall.jl` wrapper around Hugging Face tokenizers, or
  - a native tokenizer library if parity is good enough early.
- Recreate the dataset pipeline only after model parity exists.
- Treat distributed training and exact FSDP feature parity as later milestones.

Status: tokenizer bridge done; dataset/training path still pending.

### Phase 6: training recipe

- Translate the 3B FineWeb-Edu script into a Julia training script only after the model is stable.
- Keep the first Julia training target much smaller than the Python 3B recipe.
- Design checkpointing and resume behavior in Julia-native terms rather than imitating PyTorch internals too literally.

Status: pending.

## Test translation plan

The Python suite suggests this translation order:

| Python source | Julia target |
| --- | --- |
| `tests/test_main.py` RoPE and RMSNorm checks | `test_norms.jl`, `test_rope.jl` |
| `tests/test_main.py` block/model checks | `test_attention_*.jl`, `test_moe.jl`, `test_recurrent.jl`, `test_model.jl` |
| `tests/test_tokenizer.py` | `test_tokenizer.jl` |
| `tests/test_rope_debug.py` | optional Julia debug harness, not a first-class test |

## Explicit deferrals

These should not block the first Julia milestone:

- `open_mythos/moda.py`
- benchmark scripts in `tests/`
- full-scale distributed training parity
- Flash Attention-specific optimization work

## First implementation slice

If work starts immediately, the best first vertical slice is:

1. package scaffold,
2. config + RMSNorm + RoPE,
3. invariant tests,
4. GQA path,
5. recurrent block,
6. small end-to-end forward pass on a tiny config.
