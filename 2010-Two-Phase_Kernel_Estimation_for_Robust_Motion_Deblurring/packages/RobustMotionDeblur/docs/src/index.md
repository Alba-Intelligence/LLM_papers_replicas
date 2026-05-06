# RobustMotionDeblur.jl

Julia replication of Xu & Jia (ECCV 2010) two-phase kernel estimation and TV-L1 deconvolution.

## Quick start (blind deblurring)

```julia
using RobustMotionDeblur
using Images, FileIO

B = load("scripts/test1.jpg")    # or your own blurred image
I, k = deblur(B; kernel_size=25)

save("deblurred.png", clamp01nan.(I))
```

The output `I` is grayscale: the implementation estimates a single
kernel and latent luminance channel, which is usually sufficient for
evaluating algorithmic behaviour.

## Synthetic validation

For reproducible tests with a known ground-truth kernel:

```julia
using RobustMotionDeblur, Images, FileIO

sharp = load("scripts/test_pattern.png")
B, k_true = make_synthetic_blur(sharp; len=15)
I_est, k_est = deblur(B; kernel_size=15)

save("synthetic_blurred.png", clamp01nan.(B))
save("synthetic_deblurred.png", clamp01nan.(I_est))
```
