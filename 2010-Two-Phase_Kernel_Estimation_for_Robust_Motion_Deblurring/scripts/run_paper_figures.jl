#!/usr/bin/env julia

using RobustMotionDeblur
using Images, FileIO

# Placeholder script: load an external blurred image and run deblurring.

if length(ARGS) < 2
    println("Usage: run_paper_figures.jl BLURRED_IN OUTPUT_PREFIX")
    exit(1)
end

Bin = load(ARGS[1])
Ihat, kest = deblur(Bin; kernel_size=25)

save(string(ARGS[2], "_deblurred.png"), Ihat)
