# AGENTS.md

## Scope and precedence

- This file is the repository-wide source of truth for agent instructions.
- Nested `AGENTS.md` files override this file within their subtree.
- Work under `docs/private/llm_knowledge/` must also follow
  `docs/private/llm_knowledge/AGENTS.md`.
- `CLAUDE.md` files in this repository are compatibility shims only. They do
  not replace the `AGENTS.md` hierarchy.

## Project snapshot

Julia replication of Xu & Jia (ECCV 2010), *Two-Phase Kernel Estimation for
Robust Motion Deblurring*.

This is a research and teaching codebase rather than a production image
restoration library. The implementation stays close to the paper while using
practical numerical safeguards for modern Julia and Images.jl tooling.

## Stack summary

- Language: Julia
- Core libraries: `Images`, `ImageFiltering`, `FFTW`, `FileIO`, `Downloads`
- Docs: Documenter.jl under `packages/RobustMotionDeblur/docs/`
- Package layout:
  - `packages/TVL1Deconv/` for kernel-estimation internals and TV-L1 solving
  - `packages/RobustMotionDeblur/` for the public API, synthetic helpers,
    paper-data utilities, and visualization helpers

## Architecture summary

- `packages/TVL1Deconv/src/TVL1Deconv.jl` is the low-level module entry point.
- `packages/TVL1Deconv/src/edges.jl` contains the shock filter, gradient
  confidence `r`, mask `M`, and selective edge construction.
- `packages/TVL1Deconv/src/phase1.jl` implements coarse-to-fine Phase 1 kernel
  initialization (paper Eq. 6, Eq. 8, Algorithm 1).
- `packages/TVL1Deconv/src/phase2.jl` implements the Phase 2 ISD-style kernel
  refinement path and the package-level kernel-estimation wrapper.
- `packages/TVL1Deconv/src/solver.jl` contains the TV-L1 deconvolution solver
  (paper Algorithm 3), including replicate padding and explicit kernel
  centering to avoid quadrant swaps.
- `packages/RobustMotionDeblur/src/RobustMotionDeblur.jl` exposes the high-level
  public API: `deblur`, `estimate_kernel`, and `deconvolve`.
- `packages/RobustMotionDeblur/src/synthetic.jl` contains synthetic motion-kernel
  and blur-generation helpers.
- `packages/RobustMotionDeblur/src/data.jl` contains paper-data download helpers.
- `packages/RobustMotionDeblur/src/visualize.jl` contains the composite summary
  image helper.
- `scripts/` contains example and reproduction-oriented entry points.

## Where to edit

- Phase 1 kernel initialization: `packages/TVL1Deconv/src/phase1.jl`
- Edge selection, `r`, and `M`: `packages/TVL1Deconv/src/edges.jl`
- Phase 2 ISD refinement: `packages/TVL1Deconv/src/phase2.jl`
- TV-L1 solver and padding/centering: `packages/TVL1Deconv/src/solver.jl`
- High-level deblurring pipeline: `packages/RobustMotionDeblur/src/RobustMotionDeblur.jl`
- Synthetic data generation: `packages/RobustMotionDeblur/src/synthetic.jl`
- Data download logic: `packages/RobustMotionDeblur/src/data.jl`

## Repo conventions

- Public API functions should have docstrings. Internal helpers can stay
  undocumented when their role is obvious from context.
- Tests live in `packages/*/test/runtests.jl`.
- The package docs are built with:

```bash
julia --project=packages/RobustMotionDeblur/docs packages/RobustMotionDeblur/docs/make.jl
```

- Preserve paper equation references in comments when modifying algorithmic
  code. The authoritative paper is `docs/pdfs/robust_motion_deblurring.pdf`.

## Critical behavioural notes

- **Colour vs grayscale**: the heavy numerical work is intentionally done in
  luminance space. Do not describe the current pipeline as full-colour
  deblurring unless the implementation actually changes to support it.
- **Kernel alignment**: kernel estimation and deconvolution are meant to agree
  on FFT shifts and kernel centering. If quadrant artefacts reappear, inspect
  `crop_kernel_center`, `center_kernel`, `refine_kernel_isd`,
  `coarse_image_from_kernel`, and `_pad_kernel` / FFT usage in the solver.
- **Artefact tuning**: moire, ripples, and oversmoothing are mainly controlled
  by TV-L1 parameters (`λ`, `max_outer`, `max_inner`) plus any smoothing or
  thresholding applied around Phase 1 and Phase 2 kernel updates.

## Contribution guidance

- Prefer changes that preserve the current research-oriented API shape.
- Keep documentation aligned with behaviour, especially around grayscale output,
  kernel centering, and current deviations from the paper.
- Do not introduce `.github/copilot-instructions.md` unless the repository later
  gains a concrete reason to keep that format in addition to this file.

## Wiki routing

- The AI-maintained knowledge base lives under `docs/private/llm_knowledge/`.
- Treat `docs/private/llm_knowledge/raw/` as immutable copied source material.
- Treat `docs/private/llm_knowledge/wiki/` as the authored knowledge layer.
- Any work in that subtree must follow the stricter wiki-specific rules in
  `docs/private/llm_knowledge/AGENTS.md`.
