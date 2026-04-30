# Julia Reimplementation Plan

## Goal

Grow this repository as a shared Julia workspace that can host multiple model families without losing the architectural core of either one.

## Current implementation status

The workspace now has:

- `OpenMythos.jl/` for the original recurrent package,
- `DeepSeekv4.jl/` for the new DeepSeek V4 package,
- `TransformerCore.jl/` for shared low-level primitives,
- `docs/wiki/` as shared workspace documentation.

The OpenMythos package includes a parity-tested core model stack:

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
- `MythosTokenizer`
- `WarmupCosineSchedule`
- token chunking / next-token batching helpers
- `LuxHeadOnlyOpenMythos`
- `HeadOnlyTrainerState`
- resumable checkpoint helpers
- `scripts/train_3b_fineweb_edu.jl`
- `loop_index_embedding`
- `LoRAAdapter`
- `LTIInjection`
- `ACTHalting`

The current state is:

- `OpenMythos.jl/` is the recurrent-depth package with parity-oriented model code and a head-only bootstrap training path.
- `DeepSeekv4.jl/` is an architecture-first package with CSA/HCA, mHC, MoE routing, MTP, chunked-prefill-aware generation, tiny-config tests, and a head-only bootstrap training path.
- `TransformerCore.jl/` holds the shared primitive, training-utility, and first runtime-envelope layer.

The next milestone is to deepen full-model training and runtime internals without prematurely forcing all model internals into Lux layers.

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
docs/
  wiki/
```

## Phase plan

### Phase 1: workspace split

- move the original package into `OpenMythos.jl/`,
- create `DeepSeekv4.jl/`,
- keep `docs/wiki/` shared at the root.

Status: done.

### Phase 2: shared-core extraction

- move architecture-agnostic helpers into `TransformerCore.jl/`,
- keep only truly reusable primitives there,
- rewire both model packages to consume the shared package.

Status: done.

### Phase 3: OpenMythos parity-first implementation

- port the recurrent model stack,
- keep tokenizer and batching parity pragmatic,
- establish the current head-only Lux-backed bootstrap trainer.

Status: done for the current bootstrap milestone.

### Phase 4: DeepSeek V4 architecture-first implementation

- add `DeepSeekV4Config`,
- implement CSA/HCA hybrid attention,
- implement mHC residual mixing,
- implement routed/shared/hash MoE surfaces,
- implement MTP heads and cache-aware generation smoke paths.

Status: done for the current tiny-config architecture-first milestone.

### Phase 5: training foundation

- generalize the current training helpers into model-family-neutral entry points,
- define a shared loss/checkpoint/batch story that does not assume recurrent OpenMythos internals,
- land the first trainable DeepSeek milestone,
- keep the initial surface simple enough to remain readable and testable.

Status: done.

### Phase 6: advanced systems work

- first long-context runtime slice:
  - `KVCacheEnvelope`,
  - cache save/load,
  - chunked prefill,
  - envelope-aware generation for both model families,
- later work:
  - Muon and hybrid ZeRO,
  - contextual or expert parallelism,
  - deterministic fused kernels,
  - FP4 quantization-aware training,
  - lower-allocation and production-scale long-context serving.

Status: in progress; the first runtime-foundation slice is done.

## Validation strategy

| Area | Validation surface |
| --- | --- |
| `TransformerCore.jl` | primitive smoke and invariant tests |
| `OpenMythos.jl` | translated parity tests plus tokenizer/training smoke tests |
| `DeepSeekv4.jl` | tiny-config architecture, generation, and head-only training smoke tests |

## Explicit deferrals

These should not block the current Julia milestone:

- `open_mythos/moda.py`
- benchmark scripts in `tests/`
- full-scale distributed training parity
- Flash Attention-specific optimization work
- DeepSeek production runtime features beyond the current cache-envelope reference seam

## Immediate next slice

The best next vertical slice is:

1. expand beyond head-only optimization without collapsing the current shared seams,
2. decide whether lower-allocation cache internals or runtime/distribution work should land next,
3. preserve the shared runtime envelope while deeper serving work remains model-specific underneath.
