# API

The public surface of `RobustMotionDeblur` is intentionally small:

- `deblur(B; kernel_size=25, kwargs...)` – blind deblurring:
  runs Phase 1 + Phase 2 kernel estimation on `B` and then TV-L1
  deconvolution. Returns `(I, k̂)`. Important keywords:
  - `kernel_size` – support of the blur kernel.
  - `λ`, `max_outer`, `max_inner` – forwarded to TV-L1
    (overriding the built-in scale-aware defaults).
- `estimate_kernel(B; kernel_size=25, kwargs...)` – kernel estimation
  only, returning the refined kernel.
- `deconvolve(B, k; λ=2e-2, kwargs...)` – non-blind deconvolution
  of `B` given a known `k`, using the same TV-L1 backend as `deblur`.
- `make_synthetic_blur(I; len=15, θ=0.0, ...)` – helper to generate
  `(B, k_true)` by convolving a sharp input with a motion kernel and
  optional noise.

For full signatures and keyword arguments, see the generated API
reference:

```@autodocs
Modules = [RobustMotionDeblur]
Order = [:function]
```
