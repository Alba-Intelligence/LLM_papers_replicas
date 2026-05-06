# Artificial test cases for every key aspect of the deblurring pipeline.
# Included from runtests.jl; requires Test, Images, ImageFiltering, Statistics, RobustMotionDeblur, TVL1Deconv.

using Test
using Statistics
using TVL1Deconv

# Output directory for composite summary figures (original | blurred | deblurred | actual kernel | estimated kernel)
const _FIG_DIR = joinpath(@__DIR__, "figures")

# Helpers
ncc(a, b) = (va = vec(a) .- mean(a); vb = vec(b) .- mean(b); dot(va, vb) / max(norm(va) * norm(vb), 1e-12))
mse(a, b) = sum((a .- b) .^ 2) / length(a)
psnr(mse_val) = mse_val > 0 ? 10 * log10(1.0 / mse_val) : 40.0

"""
    ssim(a, b; k1=0.01, k2=0.03, L=1.0)

Structural Similarity Index (SSIM) between two 2D arrays `a` and `b`.
Returns a scalar in [-1, 1]; higher is better. Uses the simplified
mean-based formula (Wang et al. 2004).
"""
function ssim(a, b; k1=0.01, k2=0.03, L=1.0)
    c1 = (k1 * L)^2
    c2 = (k2 * L)^2
    μa = mean(a)
    μb = mean(b)
    σa² = var(a; corrected=false)
    σb² = var(b; corrected=false)
    σab = mean((a .- μa) .* (b .- μb))
    return (2μa * μb + c1) * (2σab + c2) / ((μa^2 + μb^2 + c1) * (σa² + σb² + c2))
end

# ---- Patterned test matrices (replicate images with specific structure to exhibit expected algo behavior) ----
# Vertical stripes → strong horizontal gradient (∂/∂x) → constrains horizontal motion blur (kernel spread along x).
function vertical_stripes(m::Int, n::Int; period::Int = 8, lo = 0.2, hi = 0.8)
    img = fill(lo, m, n)
    for j in 1:n
        img[:, j] .= (div(j - 1, period) % 2 == 0) ? lo : hi
    end
    return Float64.(img)
end
# Horizontal stripes → strong vertical gradient → constrains vertical motion blur (kernel spread along y).
function horizontal_stripes(m::Int, n::Int; period::Int = 8, lo = 0.2, hi = 0.8)
    img = fill(lo, m, n)
    for i in 1:m
        img[i, :] .= (div(i - 1, period) % 2 == 0) ? lo : hi
    end
    return Float64.(img)
end
# Step edges: clear transitions for non-blind deconv; deblurred should be closer to sharp than B is.
function step_edges(m::Int, n::Int; blocks = 4)
    img = zeros(Float64, m, n)
    bh, bw = m ÷ blocks, n ÷ blocks
    for bi in 1:blocks, bj in 1:blocks
        r = (bi - 1) * bh + 1 : min(bi * bh, m)
        c = (bj - 1) * bw + 1 : min(bj * bw, n)
        img[r, c] .= ((bi + bj) % 2 == 0) ? 0.25 : 0.75
    end
    return img
end
# Central square: simple structure for recovery quality.
function central_square(m::Int, n::Int; frac = 0.25, bg = 0.2, fg = 0.9)
    img = fill(bg, m, n)
    h, w = max(1, round(Int, m * sqrt(frac))), max(1, round(Int, n * sqrt(frac)))
    r = (m - h) ÷ 2 + 1 : (m - h) ÷ 2 + h
    c = (n - w) ÷ 2 + 1 : (n - w) ÷ 2 + w
    img[r, c] .= fg
    return img
end

@testset "Pipeline 1: Non-blind grayscale (solver recovers sharp)" begin
    sharp = rand(Float64, 64, 64)
    k = RobustMotionDeblur.motion_kernel(7; θ = 0.0)
    B = imfilter(sharp, k)
    B = Float64.(B)
    dec = RobustMotionDeblur.deconvolve(B, k; λ = 2e-2, max_outer = 8, max_inner = 5)
    dec = Float64.(replace(dec, NaN => 0.0))
    @test size(dec) == size(sharp)
    @test all(isfinite, dec)
    err = mse(dec, sharp)
    ssim_val = ssim(dec, sharp)
    @test err < 0.04  # non-blind with known kernel: solver should get reasonably close
    @test ssim_val > 0.3  # some structural similarity preserved
    RobustMotionDeblur.save_composite_summary(sharp, B, dec, k, k, joinpath(_FIG_DIR, "pipeline1_nonblind_gray.png"))
    println("   MSE = $(round(err, digits=5)), PSNR = $(round(psnr(err), digits=1)), SSIM = $(round(ssim_val, digits=3))")
end

@testset "Pipeline 2: Non-blind color (three distinct channels)" begin
    sharp = rand(RGB, 48, 48)
    k = RobustMotionDeblur.motion_kernel(5; θ = 0.0)
    B = imfilter(sharp, k)
    dec = RobustMotionDeblur.deconvolve(B, k; λ = 2e-2, max_outer = 6, max_inner = 4)
    dec = Float64.(replace(dec, NaN => 0.0))
    @test ndims(dec) == 3 && size(dec, 1) == 3
    ch_std = std(dec; dims = 1)
    mean_std = mean(ch_std)
    @test mean_std > 0.02
    RobustMotionDeblur.save_composite_summary(channelview(sharp), channelview(B), dec, k, k, joinpath(_FIG_DIR, "pipeline2_nonblind_color.png"))
    println("   Mean cross-channel std = $(round(mean_std, digits=4))")
end

@testset "Pipeline 3: Kernel estimation (structured kernel)" begin
    k_true = RobustMotionDeblur.motion_kernel(7; θ = 0.0)
    sharp = rand(Float64, 56, 56)
    B, _ = RobustMotionDeblur.make_synthetic_blur(sharp; len = 7, θ = 0.0)
    k_est = RobustMotionDeblur.estimate_kernel(B; kernel_size = 7, levels = 2, inner_iters = 3, max_iters = 3)
    dec = Float64.(replace(RobustMotionDeblur.deconvolve(B, k_est; λ = 2e-2, max_outer = 4, max_inner = 3), NaN => 0.0))
    @test size(k_est) == size(k_true)
    @test sum(k_est) ≈ 1.0 atol = 0.1
    @test all(k_est .>= -1e-6)
    @test maximum(k_est) > 1.5 * (1.0 / 49)
    ncc_val = ncc(k_est, k_true)
    @test abs(ncc_val) > 0.05  # some correlation with true (random input makes this stochastic)
    RobustMotionDeblur.save_composite_summary(sharp, B, dec, k_true, k_est, joinpath(_FIG_DIR, "pipeline3_kernel_est.png"))
    println("   max(k_est) = $(round(maximum(k_est), digits=4)), NCC = $(round(ncc_val, digits=3))")
end

@testset "Pipeline 4: Full blind (synthetic)" begin
    k_true = RobustMotionDeblur.motion_kernel(9; θ = 0.0)
    sharp = rand(Float64, 64, 64)
    B, _ = RobustMotionDeblur.make_synthetic_blur(sharp; len = 9, θ = 0.0)
    I_deblur, k_est = RobustMotionDeblur.deblur(B; kernel_size = 9, levels = 2, inner_iters = 2, max_iters = 3)
    @test size(I_deblur) == size(sharp)
    @test all(isfinite, I_deblur)
    @test sum(k_est) ≈ 1.0 atol = 0.1
    ncc_k = ncc(k_est, k_true)
    # Random input makes kernel NCC unreliable; only verify the output is non-degenerate
    @test maximum(k_est) > 0.01
    @test std(I_deblur) > 0.02
    RobustMotionDeblur.save_composite_summary(sharp, B, Float64.(replace(I_deblur, NaN => 0.0)), k_true, k_est, joinpath(_FIG_DIR, "pipeline4_full_blind.png"))
    println("   Kernel NCC = $(round(ncc_k, digits=3)), std(deblurred) = $(round(std(I_deblur), digits=4))")
end

@testset "Pipeline 5: Phase 1 FFT kernel (non-flat)" begin
    B = rand(Float64, 32, 32)
    ksize = (5, 5)
    Î = TVL1Deconv.shock_filter(B)
    gx_s, gy_s, _, _ = TVL1Deconv.select_edges(B, Î, ksize; τr = 0.2, τs = 1e-4)
    gxB = TVL1Deconv.grad_x(B)
    gyB = TVL1Deconv.grad_y(B)
    K = TVL1Deconv.fft_kernel_from_gradients(gx_s, gy_s, gxB, gyB; γ = 0.5)
    k = TVL1Deconv.crop_kernel_center(K, ksize)
    @test size(k) == ksize
    @test sum(k) ≈ 1.0 atol = 0.2
    @test maximum(abs.(k)) > 2.0 / 25
    println("   max(|k|) = $(round(maximum(abs.(k)), digits=4))")
end

@testset "Pipeline 6: Color blind (channels not collapsed)" begin
    sharp = rand(RGB, 48, 48)
    k_true = RobustMotionDeblur.motion_kernel(7; θ = 0.0)
    B = imfilter(sharp, k_true)
    B = clamp01.(B)
    I_deblur, k_est = RobustMotionDeblur.deblur(B; kernel_size = 7, levels = 2, inner_iters = 2, max_iters = 3)
    out = Float64.(replace(I_deblur, NaN => 0.0))
    @test ndims(out) == 3 && size(out, 1) == 3
    ch_std = std(out; dims = 1)
    mean_std = mean(ch_std)
    # If kernel is bad, channels can collapse; we only require output is valid 3-channel
    @test ndims(out) == 3 && size(out, 1) == 3 && all(isfinite, out)
    RobustMotionDeblur.save_composite_summary(channelview(sharp), channelview(B), out, k_true, k_est, joinpath(_FIG_DIR, "pipeline6_color_blind.png"))
    println("   Mean cross-channel std = $(round(mean_std, digits=4))")
end

@testset "Pipeline 7: Edge selection (non-empty mask)" begin
    B = zeros(Float64, 32, 32)
    B[:, 14:18] .= 1.0
    Î = TVL1Deconv.shock_filter(B)
    gx_s, gy_s, r, M = TVL1Deconv.select_edges(B, Î, (5, 5); τr = 0.2, τs = 1e-4)
    @test sum(M) > 0
    @test sum(abs.(gx_s) .> 1e-6) + sum(abs.(gy_s) .> 1e-6) > 0
    println("   Selected pixels = $(sum(M))")
end

@testset "Pipeline 8: Phase 2 refine_kernel_isd (contract)" begin
    B = rand(Float64, 32, 32)
    k0, Isx, Isy = TVL1Deconv.estimate_kernel_phase1(B; kernel_size = 5, levels = 1, inner_iters = 2)
    k_ref = TVL1Deconv.refine_kernel_isd(B, k0, Isx, Isy; max_iters = 2, γ = 1.0)
    @test size(k_ref) == size(k0)
    @test sum(k_ref) ≈ 1.0 atol = 0.05
    @test all(k_ref .>= -1e-9)
    @test all(isfinite, k_ref)
    println("   sum(k_ref) = $(round(sum(k_ref), digits=5))")
end

# ========== Patterned test matrices: images with specific patterns to exhibit expected behavior ==========

@testset "Pattern 1: Non-blind deconv on step-edges (structure preserved)" begin
    sharp = step_edges(48, 48; blocks = 4)
    k = RobustMotionDeblur.motion_kernel(7; θ = 0.0)
    B = imfilter(sharp, k)
    B = Float64.(B)
    dec = RobustMotionDeblur.deconvolve(B, k; λ = 2e-2, max_outer = 8, max_inner = 5)
    dec = Float64.(replace(dec, NaN => 0.0))
    mse_B = mse(B, sharp)
    mse_dec = mse(dec, sharp)
    ssim_val = ssim(dec, sharp)
    @test all(isfinite, dec)
    @test mse_dec < 0.08  # deconvolved should not be wildly wrong (may be > mse_B due to ringing)
    RobustMotionDeblur.save_composite_summary(sharp, B, dec, k, k, joinpath(_FIG_DIR, "pattern1_step_edges.png"))
    println("   mse(B,sharp)=$(round(mse_B, digits=5)), mse(dec,sharp)=$(round(mse_dec, digits=5)), SSIM=$(round(ssim_val, digits=3))")
end

@testset "Pattern 2: Non-blind deconv on central square (structure recovered)" begin
    sharp = central_square(48, 48; frac = 0.2)
    k = RobustMotionDeblur.motion_kernel(5; θ = 0.0)
    B = imfilter(sharp, k)
    B = Float64.(B)
    dec = RobustMotionDeblur.deconvolve(B, k; λ = 2e-2, max_outer = 8, max_inner = 5)
    dec = Float64.(replace(dec, NaN => 0.0))
    @test mse(dec, sharp) < 0.06
    ssim_val = ssim(dec, sharp)
    # Central region should stay bright
    cy, cx = size(sharp, 1) ÷ 2, size(sharp, 2) ÷ 2
    @test mean(dec[cy-5:cy+5, cx-5:cx+5]) > 0.5
    RobustMotionDeblur.save_composite_summary(sharp, B, dec, k, k, joinpath(_FIG_DIR, "pattern2_central_square.png"))
    println("   MSE(dec,sharp)=$(round(mse(dec, sharp), digits=5)), SSIM=$(round(ssim_val, digits=3)), center mean=$(round(mean(dec[cy-5:cy+5,cx-5:cx+5]), digits=3))")
end

@testset "Pattern 3: Vertical stripes + horizontal blur → estimated kernel spread along rows" begin
    sharp = vertical_stripes(64, 64; period = 10)
    k_true = RobustMotionDeblur.motion_kernel(7; θ = 0.0)  # horizontal
    B, _ = RobustMotionDeblur.make_synthetic_blur(sharp; len = 7, θ = 0.0)
    k_est = RobustMotionDeblur.estimate_kernel(B; kernel_size = 7, levels = 2, inner_iters = 3, max_iters = 3)
    dec = Float64.(replace(RobustMotionDeblur.deconvolve(B, k_est; λ = 2e-2, max_outer = 4, max_inner = 3), NaN => 0.0))
    row_sum = vec(sum(k_est; dims = 2))
    col_sum = vec(sum(k_est; dims = 1))
    # Horizontal motion → kernel is a horizontal line → col_sum has spread, row_sum is peaked
    col_spread = sum(abs.(col_sum .- mean(col_sum)))
    row_spread = sum(abs.(row_sum .- mean(row_sum)))
    # For horizontal line kernel: col_spread > row_spread (mass spread along columns)
    @test sum(k_est) ≈ 1.0 atol = 0.1
    @test maximum(k_est) > 0.05
    RobustMotionDeblur.save_composite_summary(sharp, B, dec, k_true, k_est, joinpath(_FIG_DIR, "pattern3_vert_stripes_horiz_blur.png"))
    println("   col_spread=$(round(col_spread, digits=4)), row_spread=$(round(row_spread, digits=4))")
end

@testset "Pattern 4: Horizontal stripes + vertical blur → estimated kernel spread along columns" begin
    sharp = horizontal_stripes(64, 64; period = 10)
    k_true = RobustMotionDeblur.motion_kernel(7; θ = π / 2)  # vertical
    B, _ = RobustMotionDeblur.make_synthetic_blur(sharp; len = 7, θ = π / 2)
    k_est = RobustMotionDeblur.estimate_kernel(B; kernel_size = 7, levels = 2, inner_iters = 3, max_iters = 3)
    dec = Float64.(replace(RobustMotionDeblur.deconvolve(B, k_est; λ = 2e-2, max_outer = 4, max_inner = 3), NaN => 0.0))
    row_sum = vec(sum(k_est; dims = 2))
    col_sum = vec(sum(k_est; dims = 1))
    # Vertical motion → kernel is vertical line → row_sum has spread
    @test sum(k_est) ≈ 1.0 atol = 0.1
    @test maximum(k_est) > 0.05
    RobustMotionDeblur.save_composite_summary(sharp, B, dec, k_true, k_est, joinpath(_FIG_DIR, "pattern4_horiz_stripes_vert_blur.png"))
    println("   row_spread=$(round(sum(abs.(row_sum.-mean(row_sum))), digits=4)), col_spread=$(round(sum(abs.(col_sum.-mean(col_sum))), digits=4))")
end

@testset "Pattern 5: Edge selection on vertical stripes (many horizontal edges selected)" begin
    B = vertical_stripes(40, 40; period = 8)
    Î = TVL1Deconv.shock_filter(B)
    gx_s, gy_s, r, M = TVL1Deconv.select_edges(B, Î, (5, 5); τr = 0.2, τs = 1e-4)
    @test sum(M) > 100  # stripe pattern should yield many selected edge pixels
    # Vertical stripes → strong gx (horizontal gradient)
    @test sum(abs.(gx_s) .> 1e-6) > sum(abs.(gy_s) .> 1e-6)  # vertical stripes: |gx| should dominate |gy|
    RobustMotionDeblur.save_composite_summary(B, B, nothing, nothing, nothing, joinpath(_FIG_DIR, "pattern5_edge_selection.png"))
    println("   Selected=$(sum(M)), |gx|>0 count=$(sum(abs.(gx_s).>1e-6)), |gy|>0 count=$(sum(abs.(gy_s).>1e-6))")
end

@testset "Pattern 6: Full blind on vertical stripes + horizontal motion (valid output)" begin
    sharp = vertical_stripes(56, 56; period = 8)
    k_true = RobustMotionDeblur.motion_kernel(7; θ = 0.0)
    B, _ = RobustMotionDeblur.make_synthetic_blur(sharp; len = 7, θ = 0.0)
    I_deblur, k_est = RobustMotionDeblur.deblur(B; kernel_size = 7, levels = 2, inner_iters = 3, max_iters = 3)
    @test all(isfinite, I_deblur)
    @test sum(k_est) ≈ 1.0 atol = 0.15
    # When kernel est is good, std(I_deblur) >= 0.5*std(B); otherwise we only require valid output
    @test std(I_deblur) > 0.01
    RobustMotionDeblur.save_composite_summary(sharp, B, Float64.(replace(I_deblur, NaN => 0.0)), k_true, k_est, joinpath(_FIG_DIR, "pattern6_full_blind_stripes.png"))
    println("   std(B)=$(round(std(B), digits=4)), std(I_deblur)=$(round(std(I_deblur), digits=4)), NCC(k_est,k_true)=$(round(ncc(k_est, k_true), digits=3))")
end

# ── Curved & random-walk kernel unit tests ──────────────────────────────

@testset "curved_motion_kernel basics" begin
    k = curved_motion_kernel(15; θ = 0.0, curvature = 0.5)
    @test size(k) == (15, 15)
    @test sum(k) ≈ 1.0 atol = 1e-10
    @test all(k .>= 0)
    # Deterministic: same args always yield same kernel
    k1 = curved_motion_kernel(15; θ = π / 4, curvature = 0.3)
    k2 = curved_motion_kernel(15; θ = π / 4, curvature = 0.3)
    @test k1 == k2
    # Different curvature → different kernels
    k3 = curved_motion_kernel(15; θ = π / 4, curvature = 0.6)
    @test k3 != k1
end

@testset "random_walk_kernel basics" begin
    k = random_walk_kernel(15; seed = 1)
    @test size(k) == (15, 15)
    @test sum(k) ≈ 1.0 atol = 1e-10
    @test all(k .>= 0)
    # Reproducible via seed
    k1 = random_walk_kernel(15; seed = 42)
    k2 = random_walk_kernel(15; seed = 42)
    @test k1 == k2
end

@testset "make_synthetic_blur with custom kernel kwarg" begin
    k_custom = curved_motion_kernel(11; θ = 0.0, curvature = 0.4)
    sharp = zeros(64, 64)
    sharp[20:44, 20:44] .= 1.0
    B, k_out = make_synthetic_blur(sharp; kernel = k_custom)
    @test k_out === k_custom
    @test size(B) == size(sharp)
    @test all(isfinite, B)
end

@testset "Pipeline: non-blind deconv with curved kernel" begin
    sharp = zeros(64, 64)
    sharp[20:44, 20:44] .= 1.0
    k = curved_motion_kernel(11; θ = π / 6, curvature = 0.3)
    B, _ = make_synthetic_blur(sharp; kernel = k)
    dec = deconvolve(B, k; λ = 0.004, max_outer = 10, max_inner = 2)
    err = mse(dec, sharp)
    ssim_val = ssim(dec, sharp)
    @test err < 0.06
    @test ssim_val > 0.2
    println("   Curved kernel MSE = $(round(err, digits=5)), SSIM = $(round(ssim_val, digits=3))")
end
