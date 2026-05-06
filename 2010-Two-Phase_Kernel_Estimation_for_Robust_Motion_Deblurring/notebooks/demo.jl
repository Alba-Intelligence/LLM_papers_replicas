### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 8a1b2c3d-0001-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "packages", "RobustMotionDeblur"))
    using RobustMotionDeblur
    using Images
    using Statistics
end

# ╔═╡ 8a1b2c3d-0002-4000-8000-000000000001
md"""
# Robust Motion Deblurring Demo

This notebook demonstrates the two-phase kernel estimation pipeline from
**Xu & Jia (ECCV 2010)** for blind motion deblurring.

We will:
1. Create a sharp test image
2. Blur it with different motion kernels (linear, curved, random-walk)
3. Run blind and non-blind deblurring
4. Compare results
"""

# ╔═╡ 8a1b2c3d-0003-4000-8000-000000000001
md"## 1. Create a sharp test image"

# ╔═╡ 8a1b2c3d-0004-4000-8000-000000000001
begin
    # Simple synthetic image: bright square on dark background with some texture
    sharp = zeros(Float64, 128, 128)
    sharp[30:98, 30:98] .= 0.9
    # Add a gradient stripe for visible deblur quality
    for i in 40:88
        sharp[i, 40:88] .= 0.4 + 0.5 * (i - 40) / 48
    end
    Gray.(sharp)
end

# ╔═╡ 8a1b2c3d-0005-4000-8000-000000000001
md"## 2. Generate motion blur kernels"

# ╔═╡ 8a1b2c3d-0006-4000-8000-000000000001
begin
    k_linear = motion_kernel(15; θ = π / 6)
    k_curved = curved_motion_kernel(15; θ = 0.0, curvature = 0.25)
    k_random = random_walk_kernel(15; seed = 42)
    
    # Visualize kernels (scale for visibility)
    pad = zeros(size(k_linear, 1), 3)
    kernel_strip = hcat(
        k_linear ./ maximum(k_linear),
        pad,
        k_curved ./ maximum(k_curved),
        pad,
        k_random ./ maximum(k_random),
    )
    Gray.(kernel_strip)
end

# ╔═╡ 8a1b2c3d-0007-4000-8000-000000000001
md"""
**Left to right:** linear (θ=30°), curved (curvature=0.25), random walk (seed=42).
Each kernel is 15×15 and normalized to sum to 1.
"""

# ╔═╡ 8a1b2c3d-0008-4000-8000-000000000001
md"## 3. Apply blur and noise"

# ╔═╡ 8a1b2c3d-0009-4000-8000-000000000001
begin
    B_linear, _ = make_synthetic_blur(sharp; kernel = k_linear, noise_std = 0.005)
    B_curved, _ = make_synthetic_blur(sharp; kernel = k_curved, noise_std = 0.005)
    B_random, _ = make_synthetic_blur(sharp; kernel = k_random, noise_std = 0.005)
    
    sep = ones(128, 3)
    blurred_strip = hcat(
        Float64.(B_linear), sep,
        Float64.(B_curved), sep,
        Float64.(B_random),
    )
    Gray.(blurred_strip)
end

# ╔═╡ 8a1b2c3d-000a-4000-8000-000000000001
md"""
**Left to right:** linear blur, curved blur, random-walk blur.
All have light Gaussian noise (σ=0.005).
"""

# ╔═╡ 8a1b2c3d-000b-4000-8000-000000000001
md"## 4. Non-blind deconvolution (known kernel)"

# ╔═╡ 8a1b2c3d-000c-4000-8000-000000000001
begin
    dec_linear = deconvolve(B_linear, k_linear; λ = 0.004, max_outer = 15, max_inner = 3)
    dec_curved = deconvolve(B_curved, k_curved; λ = 0.004, max_outer = 15, max_inner = 3)
    dec_random = deconvolve(B_random, k_random; λ = 0.004, max_outer = 15, max_inner = 3)
    
    deconv_strip = hcat(
        Float64.(dec_linear), sep,
        Float64.(dec_curved), sep,
        Float64.(dec_random),
    )
    Gray.(clamp.(deconv_strip, 0, 1))
end

# ╔═╡ 8a1b2c3d-000d-4000-8000-000000000001
md"""
**Non-blind TV-L1 deconvolution** with the known ground-truth kernel.
This is the best the solver can do — no kernel estimation error.
"""

# ╔═╡ 8a1b2c3d-000e-4000-8000-000000000001
md"## 5. Blind deblurring (kernel estimated from blurred image)"

# ╔═╡ 8a1b2c3d-000f-4000-8000-000000000001
begin
    I_blind, k_est = deblur(Float64.(B_linear); kernel_size = 15, levels = 3)
    
    # Show estimated vs true kernel side by side
    k_est_vis = k_est ./ max(maximum(k_est), 1e-10)
    k_true_vis = k_linear ./ maximum(k_linear)
    kern_compare = hcat(k_true_vis, pad, k_est_vis)
    Gray.(kern_compare)
end

# ╔═╡ 8a1b2c3d-0010-4000-8000-000000000001
md"**Left:** true kernel. **Right:** estimated kernel from blind deblurring."

# ╔═╡ 8a1b2c3d-0011-4000-8000-000000000001
begin
    blind_strip = hcat(
        Float64.(B_linear), sep,
        Float64.(clamp.(I_blind, 0, 1)),
    )
    Gray.(blind_strip)
end

# ╔═╡ 8a1b2c3d-0012-4000-8000-000000000001
md"**Left:** blurred input. **Right:** blind deblurred output."

# ╔═╡ 8a1b2c3d-0013-4000-8000-000000000001
md"## 6. Quantitative comparison"

# ╔═╡ 8a1b2c3d-0014-4000-8000-000000000001
begin
    _mse(a, b) = sum((a .- b) .^ 2) / length(a)
    _psnr(m) = m > 0 ? 10 * log10(1.0 / m) : Inf
    function _ssim(a, b; L = 1.0)
        c1, c2 = (0.01L)^2, (0.03L)^2
        μa, μb = mean(a), mean(b)
        σa², σb² = var(a; corrected = false), var(b; corrected = false)
        σab = mean((a .- μa) .* (b .- μb))
        (2μa * μb + c1) * (2σab + c2) / ((μa^2 + μb^2 + c1) * (σa² + σb² + c2))
    end
    
    results = [
        ("Linear (non-blind)", dec_linear),
        ("Curved (non-blind)", dec_curved),
        ("Random (non-blind)", dec_random),
        ("Linear (blind)", I_blind),
    ]
    
    header = "| Method | MSE | PSNR (dB) | SSIM |\n|--------|-----|-----------|------|\n"
    rows = map(results) do (name, img)
        m = _mse(img, sharp)
        "| $name | $(round(m, digits=5)) | $(round(_psnr(m), digits=1)) | $(round(_ssim(img, sharp), digits=3)) |"
    end
    Markdown.parse(header * join(rows, "\n"))
end

# ╔═╡ 8a1b2c3d-0015-4000-8000-000000000001
md"""
## Notes

- All processing is done in **grayscale** (luminance space). Colour images
  are converted automatically; the output is grayscale.
- The **TV-L1 solver** uses replicate padding and explicit kernel centering
  to avoid quadrant artefacts.
- **Phase 1** uses multi-scale shock-filtered edge selection (Xu & Jia §3).
- **Phase 2** refines the kernel via ISD (§4).
- Adjust `λ` (regularisation weight) to trade off sharpness vs ringing.
"""

# ╔═╡ Cell order:
# ╟─8a1b2c3d-0002-4000-8000-000000000001
# ╠═8a1b2c3d-0001-4000-8000-000000000001
# ╟─8a1b2c3d-0003-4000-8000-000000000001
# ╠═8a1b2c3d-0004-4000-8000-000000000001
# ╟─8a1b2c3d-0005-4000-8000-000000000001
# ╠═8a1b2c3d-0006-4000-8000-000000000001
# ╟─8a1b2c3d-0007-4000-8000-000000000001
# ╟─8a1b2c3d-0008-4000-8000-000000000001
# ╠═8a1b2c3d-0009-4000-8000-000000000001
# ╟─8a1b2c3d-000a-4000-8000-000000000001
# ╟─8a1b2c3d-000b-4000-8000-000000000001
# ╠═8a1b2c3d-000c-4000-8000-000000000001
# ╟─8a1b2c3d-000d-4000-8000-000000000001
# ╟─8a1b2c3d-000e-4000-8000-000000000001
# ╠═8a1b2c3d-000f-4000-8000-000000000001
# ╟─8a1b2c3d-0010-4000-8000-000000000001
# ╠═8a1b2c3d-0011-4000-8000-000000000001
# ╟─8a1b2c3d-0012-4000-8000-000000000001
# ╟─8a1b2c3d-0013-4000-8000-000000000001
# ╠═8a1b2c3d-0014-4000-8000-000000000001
# ╟─8a1b2c3d-0015-4000-8000-000000000001
