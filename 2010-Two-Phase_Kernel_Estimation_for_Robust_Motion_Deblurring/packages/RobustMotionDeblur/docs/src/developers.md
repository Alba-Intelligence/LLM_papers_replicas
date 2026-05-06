# Developers

- Main entry points: `RobustMotionDeblur.deblur`, `estimate_kernel`,
  `deconvolve`. All three live in `src/RobustMotionDeblur.jl`.
- Kernel estimation internals live in `packages/TVL1Deconv/src/`
  (`edges.jl`, `phase1.jl`, `phase2.jl`).
- TV-L1 solver internals live in `packages/TVL1Deconv/src/solver.jl`.
- High-level scripts for experiments are in `scripts/` (see
  `synthetic_validation.jl` and `run_paper_figures.jl`).

To build the Documenter site:

```julia
julia --project=packages/RobustMotionDeblur/docs packages/RobustMotionDeblur/docs/make.jl
```
