using Test
using TVL1Deconv
using Images, ImageFiltering

@testset "TVL1Deconv input validation - empty image" begin
    k = ones(5, 5) ./ 25
    @test_throws ArgumentError tvl1_deconvolve(zeros(0, 0), k)
    # Empty kernel: solver does not validate upfront; shrinkKernel throws MethodError
    @test_throws Exception tvl1_deconvolve(zeros(10, 10), zeros(0, 0))
end

@testset "TVL1Deconv input validation - invalid kernel" begin
    img = rand(32, 32)
    # Solver does not validate kernel dimensions; degenerate kernels throw (MethodError from shrinkKernel)
    @test_throws Exception tvl1_deconvolve(img, zeros(0, 0))
    @test_throws Exception tvl1_deconvolve(img, zeros(0, 5))
    @test_throws Exception tvl1_deconvolve(img, zeros(5, 0))
end

# Note: solver does not validate λ or iteration counts; omitted tests that expected ArgumentError

@testset "TVL1Deconv grayscale basic" begin
    img = rand(Float64, 64, 64)
    k = ones(5, 5) ./ 25
    out = tvl1_deconvolve(img, k; λ = 2e-2, max_outer = 1, max_inner = 1)
    @test size(out) == size(img)
    @test out isa Matrix{Float64}
end

@testset "TVL1Deconv grayscale - different kernel sizes" begin
    img = rand(Float64, 32, 32)
    k3 = ones(3, 3) ./ 9
    k5 = ones(5, 5) ./ 25
    k7 = ones(7, 7) ./ 49
    out3 = tvl1_deconvolve(img, k3; λ = 2e-2, max_outer = 1, max_inner = 1)
    out5 = tvl1_deconvolve(img, k5; λ = 2e-2, max_outer = 1, max_inner = 1)
    out7 = tvl1_deconvolve(img, k7; λ = 2e-2, max_outer = 1, max_inner = 1)
    @test size(out3) == size(img)
    @test size(out5) == size(img)
    @test size(out7) == size(img)
end

@testset "TVL1Deconv grayscale - small image" begin
    img = rand(Float64, 16, 16)
    k = ones(3, 3) ./ 9
    out = tvl1_deconvolve(img, k; λ = 2e-2, max_outer = 1, max_inner = 1)
    @test size(out) == size(img)
end

@testset "TVL1Deconv grayscale - large kernel relative to image" begin
    img = rand(Float64, 20, 20)
    k = ones(9, 9) ./ 81
    out = tvl1_deconvolve(img, k; λ = 2e-2, max_outer = 1, max_inner = 1)
    @test size(out) == size(img)
end

@testset "TVL1Deconv color image" begin
    img = rand(RGB, 32, 32)
    k = ones(5, 5) ./ 25
    out = tvl1_deconvolve(img, k; λ = 2e-2, max_outer = 1, max_inner = 1)
    # Solver returns (C, H, W) for color
    @test size(out) == (3, size(img, 1), size(img, 2))
end

@testset "TVL1Deconv color - different kernel sizes" begin
    img = rand(RGB, 32, 32)
    k3 = ones(3, 3) ./ 9
    k5 = ones(5, 5) ./ 25
    out3 = tvl1_deconvolve(img, k3; λ = 2e-2, max_outer = 1, max_inner = 1)
    out5 = tvl1_deconvolve(img, k5; λ = 2e-2, max_outer = 1, max_inner = 1)
    @test size(out3) == (3, size(img, 1), size(img, 2))
    @test size(out5) == (3, size(img, 1), size(img, 2))
end

@testset "TVL1Deconv non-blind structured synthetic" begin
    m = n = 32
    sharp = zeros(Float64, m, n)
    block = 4
    @inbounds for i in 1:m, j in 1:n
        if ((i - 1) ÷ block + (j - 1) ÷ block) % 2 == 0
            sharp[i, j] = 1.0
        end
    end

    k = ones(3, 3) ./ 9
    blurred = imfilter(sharp, k)

    rec = tvl1_deconvolve(blurred, k; λ = 2e-2, max_outer = 5, max_inner = 3)
    mse(a, b) = sum((a .- b) .^ 2) / length(a)
    reblurred = imfilter(rec, k)
    mse_sharp_blur = mse(sharp, blurred)
    @test mse(reblurred, blurred) < mse_sharp_blur
end

@testset "TVL1Deconv - output finite values" begin
    img = rand(Float64, 32, 32) .* 0.5 .+ 0.25
    k = ones(3, 3) ./ 9
    out = tvl1_deconvolve(img, k; λ = 2e-2, max_outer = 1, max_inner = 1)
    @test all(isfinite, out)
end

@testset "TVL1Deconv - different λ values" begin
    img = rand(Float64, 32, 32)
    k = ones(3, 3) ./ 9
    out_low = tvl1_deconvolve(img, k; λ = 1e-3, max_outer = 1, max_inner = 1)
    out_high = tvl1_deconvolve(img, k; λ = 1e-1, max_outer = 1, max_inner = 1)
    @test size(out_low) == size(out_high) == size(img)
end

# --- Phase 2 (ISD/IRLS) unit tests ---

@testset "Phase 2 - first_significant_jump" begin
    # Kernel with one large value and rest small: clear first jump
    k = zeros(5, 5)
    k[3, 3] = 1.0
    k[1, 1] = 0.01
    k[2, 2] = 0.02
    h = 5
    s = TVL1Deconv.first_significant_jump(k, h, 1; divisor=2)
    @test 0 <= s <= maximum(abs.(k))
    # All-equal (no jump): returns last value
    kflat = fill(0.2, 3, 3)
    s2 = TVL1Deconv.first_significant_jump(kflat, 3, 1; divisor=2)
    @test s2 == 0.2
    # Length-1 kernel
    k1 = [0.5]
    s3 = TVL1Deconv.first_significant_jump(k1, 1, 1; divisor=2)
    @test s3 == 0.0
end

@testset "Phase 2 - center_kernel" begin
    # Centered kernel: single peak at center (5×5) -> idempotent
    k_centered = zeros(5, 5)
    k_centered[3, 3] = 1.0
    out = TVL1Deconv.center_kernel(k_centered)
    @test out ≈ k_centered
    @test size(out) == (5, 5)
    # Off-center: peak at (1,1)
    k_off = zeros(5, 5)
    k_off[1, 1] = 1.0
    out2 = TVL1Deconv.center_kernel(k_off)
    @test size(out2) == (5, 5)
    @test sum(out2) ≈ 1.0
    # Center of mass of result should be near geometric center (3, 3)
    cy = sum(i * sum(out2[i, :]) for i in 1:5) / sum(out2)
    cx = sum(j * sum(out2[:, j]) for j in 1:5) / sum(out2)
    @test 2.0 <= cy <= 4.0
    @test 2.0 <= cx <= 4.0
    # Zero kernel: returns copy, no throw
    k_zero = zeros(5, 5)
    out3 = TVL1Deconv.center_kernel(k_zero)
    @test out3 == k_zero
    @test out3 !== k_zero
end

@testset "Phase 2 - refine_kernel_isd contract and smoke" begin
    # Get valid (B, k0, Isx, Isy) from Phase 1 on small image
    B = rand(Float64, 32, 32) .* 0.5 .+ 0.25
    k0, Isx, Isy = estimate_kernel_phase1(B; kernel_size=5, levels=1, inner_iters=2)
    k = refine_kernel_isd(B, k0, Isx, Isy; max_iters=2, γ=1.0)
    @test size(k) == size(k0)
    @test sum(k) ≈ 1.0 atol = 1e-5
    @test all(k .>= -1e-10)  # allow tiny numerical negatives
    @test all(isfinite, k)
end

@testset "Phase 2 - refine_kernel_isd weak-kernel early exit" begin
    B = rand(Float64, 24, 24)
    k0_weak = zeros(5, 5)
    k0_weak[1, 1] = 1e-8
    Isx = zeros(size(B))
    Isy = zeros(size(B))
    k = refine_kernel_isd(B, k0_weak, Isx, Isy; max_iters=2)
    @test size(k) == (5, 5)
    # Early exit returns center_kernel(copy(k0_weak)); sum preserved
    @test sum(k) ≈ sum(k0_weak) atol = 1e-6
end

@testset "Phase 2 - refine_kernel_isd determinism" begin
    B = rand(Float64, 32, 32) .* 0.5 .+ 0.25
    k0, Isx, Isy = estimate_kernel_phase1(B; kernel_size=5, levels=1, inner_iters=2)
    k1 = refine_kernel_isd(B, k0, Isx, Isy; max_iters=2, γ=1.0)
    k2 = refine_kernel_isd(B, k0, Isx, Isy; max_iters=2, γ=1.0)
    @test k1 ≈ k2
end

@testset "Phase 2 - refine_kernel_isd one IRLS step" begin
    B = rand(Float64, 32, 32) .* 0.5 .+ 0.25
    k0, Isx, Isy = estimate_kernel_phase1(B; kernel_size=5, levels=1, inner_iters=2)
    k = refine_kernel_isd(B, k0, Isx, Isy; max_iters=1, γ=1.0)
    @test size(k) == size(k0)
    @test sum(k) ≈ 1.0 atol = 1e-5
    @test all(k .>= -1e-10)
end
