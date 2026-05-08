# Julia Reimplementation Plan

## Goal

Grow this repository as a shared Julia workspace that can host multiple model families without losing the architectural core of either one.

## Current implementation status

The workspace now has:

- `OpenMythos.jl/` for the original recurrent package,
- `DeepSeekv4.jl/` for the new DeepSeek V4 package,
- `TransformerCore.jl/` for shared low-level primitives,
- `docs/wiki/` as shared workspace narrative documentation,
- `docs/` as a shared `Documenter.jl` API site.

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
- `FullModelTrainerState`
- dense full-model bootstrap configs and checkpoint helpers
- resumable checkpoint helpers
- `scripts/train_3b_fineweb_edu.jl`
- `loop_index_embedding`
- `LoRAAdapter`
- `LTIInjection`
- `ACTHalting`
- Lux-native OpenMythos layers for attention, experts / MoE, transformer blocks, recurrent update primitives, and the tied-embedding model shell
- `LuxFullModelTrainerState` built on the shared `TransformerCore.NextTokenTrainerState`
- shared Lux full-model checkpoint save/load wrappers for OpenMythos

The current state is:

- `OpenMythos.jl/` is the recurrent-depth package with parity-oriented model code, a head-only Lux bootstrap trainer, a legacy mutable full-model bootstrap trainer, and a new Lux-native `LuxFullModelTrainerState` plus shared save/load wrappers on the shared trainer foundation for small GQA/MLA configs including optional shared experts; the script default now points at the Lux-native path and keeps legacy checkpoint directories isolated.
- `DeepSeekv4.jl/` is an architecture-first package with CSA/HCA, mHC, an optional gated Engram branch, MoE routing, MTP, chunked-prefill-aware generation, tiny-config tests, a head-only bootstrap training path, and a first tiny full-model bootstrap trainer that now updates both the main LM logits path and the current auxiliary MTP heads.
- `TransformerCore.jl/` holds the shared primitive, Lux-native layer/trainer foundation, runtime-envelope, and paged/growable cache-buffer layer.
- the workspace source now carries public-API docstrings and a shared Documenter build for the multi-package surface.

The next milestone is to refactor both model families toward a clearer Lux-native full-model training story without prematurely collapsing their model-family-specific internals into one abstraction. `TransformerCore.jl` now has the first shared Lux-native layer/trainer/checkpoint foundation, and `OpenMythos.jl` now has Lux-native attention/FFN/block/recurrent mirrors, a tied-embedding `LuxOpenMythos` shell, and a first `LuxFullModelTrainerState` plus shared save/load wrappers on top of that foundation; the next package-level work is to replace the remaining legacy package training/checkpoint path and script defaults with that surface first, then reuse the same pattern in DeepSeek.

For future DeepSeek planning, keep the source priority explicit:

- the official DeepSeek V4 note is still the implementation authority,
- the Kili article plus Engram/mHC/V3 references are useful for lineage and future experiments,
- speculative pre-release analyses should guide questions, not silently redefine the target architecture.

## Chosen stack

| Concern                               | Julia library                                       |
| ------------------------------------- | --------------------------------------------------- |
| Model definition                      | `Lux.jl`                                            |
| Tensor ops and NN primitives          | `NNlib.jl`                                          |
| Autodiff                              | `Zygote.jl`                                         |
| Optimizers                            | `Optimisers.jl`                                     |
| Data loading                          | `MLUtils.jl`                                        |
| GPU execution                         | `CUDA.jl`                                           |
| Documentation                         | `Documenter.jl` + `docs/wiki`                       |
| Serialization                         | `JLD2.jl` or another Julia-native checkpoint format |
| Temporary Python interop where needed | `PythonCall.jl`                                     |

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
  - growable axis buffers replacing repeated cache concatenation in attention paths,
  - envelope-level capacity hints for preallocated cache growth,
- later work:
  - Muon and hybrid ZeRO,
  - contextual or expert parallelism,
  - deterministic fused kernels,
  - FP4 quantization-aware training,
  - paged/preallocated and production-scale long-context serving.

Status: in progress; the runtime-envelope, lower-allocation cache-buffer, preallocated-capacity, true paged-cache-buffer, first gated Engram slice, and first shared Lux-native trainer/checkpoint foundation are done.

## Validation strategy

| Area                 | Validation surface                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `TransformerCore.jl` | primitive smoke and invariant tests                                                                        |
| `OpenMythos.jl`      | translated parity tests plus tokenizer, head-only, and dense full-model training smoke tests               |
| `DeepSeekv4.jl`      | tiny-config architecture, generation, Engram, head-only training, and tiny full-model training smoke tests |

## Explicit deferrals

These should not block the current Julia milestone:

- `open_mythos/moda.py`
- benchmark scripts in `tests/`
- full-scale distributed training parity
- Flash Attention-specific optimization work
- DeepSeek production runtime features beyond the current cache-envelope reference seam

## Immediate next slice

The best next vertical slice is:

1. continue the OpenMythos Lux refactor by replacing the remaining legacy package training/checkpoint internals with the new `LuxOpenMythos` + `TransformerCore.jl` trainer/checkpoint foundation and make full-model training the primary path,
2. replace the OpenMythos Python tokenizer/FineWeb bridges with Julia-native paths where practical,
3. preserve the shared runtime envelope while page-aware attention and deeper serving work remain model-specific underneath,
4. expand distributed/runtime work once the single-process story is deeper,
5. reuse the same shared training surface inside DeepSeek after the OpenMythos refactor is stable.
