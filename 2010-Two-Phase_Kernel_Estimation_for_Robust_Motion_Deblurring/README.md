## RobustMotionDeblur.jl

Julia replication of Xu & Jia, *Two-Phase Kernel Estimation for Robust Motion Deblurring* (ECCV 2010).

- Blind kernel estimation with selective edges and ISD-style refinement.
- Robust TV-L1 non-blind deconvolution.
- Synthetic data helpers and hooks for reproducing paper-style examples.

This is a research/teaching codebase rather than a drop-in production
library. The implementation follows the paper closely but uses a few
simplifications and conservative defaults to keep it numerically stable
on modern Julia/Images.jl.

### Installation

From Julia's package REPL:

```julia
] dev path/to/2010-Two-Phase_Kernel_Estimation_for_Robust_Motion_Deblurring/packages/RobustMotionDeblur
```

Then:

```julia
using RobustMotionDeblur
```

### Quick example (blind deblurring)

```julia
using RobustMotionDeblur
using Images, FileIO

B = load("scripts/test1.jpg")           # or your own blurred image
I, k = deblur(B; kernel_size=25)

save("deblurred.png", clamp01nan.(I))
```

Internally, the blurred image is edge-tapered before kernel estimation
and deconvolution to reduce FFT ringing at the borders. The kernel is
estimated in a coarse-to-fine fashion and recentred to avoid quadrant
swaps in the output.

### Synthetic validation

```julia
using RobustMotionDeblur, Images, FileIO

sharp = load("scripts/test_pattern.png")   # any reasonably sharp image
B, k_true = make_synthetic_blur(sharp; len=15)
I_est, k_est = deblur(B; kernel_size=15)

save("synthetic_blurred.png", clamp01nan.(B))
save("synthetic_deblurred.png", clamp01nan.(I_est))
```

See `scripts/synthetic_validation.jl` and `AGENTS.md` for a more complete
pipeline description.

### Current behaviour and caveats

- **Grayscale output**: the pipeline internally converts colour to
  luminance for kernel estimation and TV-L1 deconvolution. The returned
  `I` is grayscale even if the input is RGB.
- **Boundary handling**: TV-L1 deconvolution uses FFTs with explicit
  replicate padding and careful kernel centering to reduce wrap-around
  artefacts, but you may still see mild ringing near strong edges. The
  input is also passed through a simple cosine **edge taper** to reduce
  boundary discontinuities.
- **Moiré / ripple patterns**: on strongly blurred, high-frequency
  images, an aggressive deconvolution can amplify noise and produce
  moiré-like ripples. Tuning `lambda`, `max_outer`, and `max_inner`
  (see `TVL1Deconv.tvl1_deconvolve`) trades off sharpness vs artefacts.
- **Tiny images**: for very small inputs, the default kernel size and
  regularisation are intentionally conservative; you should not expect
  strong deblurring, but the pipeline should run without quadrant swaps
  or gross misalignment.

By default, the TV-L1 parameters are chosen heuristically based on the
image size (more iterations and weaker data term for small images, fewer
iterations and a stronger data term for large ones). You can always
override these via keyword arguments to `deblur` / `deconvolve`.

The `docs/` directory of `RobustMotionDeblur` contains a more detailed
algorithm and API description built with Documenter.jl.
