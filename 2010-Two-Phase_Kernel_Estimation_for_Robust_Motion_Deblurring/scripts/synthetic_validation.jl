#!/usr/bin/env julia

using RobustMotionDeblur
using Images

# Simple synthetic validation: blur a random image and attempt to recover it.

img = rand(Float64, 128, 128)
B, ktrue = make_synthetic_blur(img; len=15)

Ihat, kest = deblur(B; kernel_size=15, levels=1, inner_iters=2, max_iters=2)

println("Synthetic validation done.")
