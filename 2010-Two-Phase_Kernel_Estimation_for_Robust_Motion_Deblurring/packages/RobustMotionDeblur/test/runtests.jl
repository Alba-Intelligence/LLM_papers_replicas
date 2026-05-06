using Test
using RobustMotionDeblur
using Images, ImageFiltering, Statistics

@testset "motion_kernel - horizontal (θ=0)" begin
    k = motion_kernel(5; θ=0.0)
    @test size(k) == (5, 5)
    @test sum(k) ≈ 1.0
    @test all(k .>= 0)
end

@testset "motion_kernel - vertical (θ=π/2)" begin
    k = motion_kernel(5; θ=π/2)
    @test size(k) == (5, 5)
    @test sum(k) ≈ 1.0
    @test all(k .>= 0)
end

@testset "motion_kernel - diagonal (θ=π/4)" begin
    k = motion_kernel(7; θ=π/4)
    @test size(k) == (7, 7)
    @test sum(k) ≈ 1.0
    @test all(k .>= 0)
end

@testset "motion_kernel - length 1" begin
    k = motion_kernel(1)
    @test size(k) == (1, 1)
    @test k[1, 1] ≈ 1.0
end

@testset "motion_kernel - length 3" begin
    k = motion_kernel(3)
    @test size(k) == (3, 3)
    @test sum(k) ≈ 1.0
end

@testset "motion_kernel - length 9" begin
    k = motion_kernel(9)
    @test size(k) == (9, 9)
    @test sum(k) ≈ 1.0
end

@testset "motion_kernel - different angles" begin
    for θ in [0.0, π/6, π/4, π/3, π/2, π]
        k = motion_kernel(7; θ=θ)
        @test size(k) == (7, 7)
        @test sum(k) ≈ 1.0
        @test all(k .>= 0)
    end
end

@testset "make_synthetic_blur - basic grayscale" begin
    sharp = rand(Float64, 64, 64)
    B, k = make_synthetic_blur(sharp; len=9)
    @test size(B) == size(sharp)
    @test size(k) == (9, 9)
    @test sum(k) ≈ 1.0
    @test all(B .>= 0)
    @test all(B .<= 1)
end

@testset "make_synthetic_blur - with horizontal motion" begin
    sharp = rand(Float64, 32, 32)
    B, k = make_synthetic_blur(sharp; len=11, θ=0.0)
    @test size(B) == size(sharp)
    @test size(k) == (11, 11)
end

@testset "make_synthetic_blur - with vertical motion" begin
    sharp = rand(Float64, 32, 32)
    B, k = make_synthetic_blur(sharp; len=11, θ=π/2)
    @test size(B) == size(sharp)
    @test size(k) == (11, 11)
end

@testset "make_synthetic_blur - with diagonal motion" begin
    sharp = rand(Float64, 32, 32)
    B, k = make_synthetic_blur(sharp; len=7, θ=π/4)
    @test size(B) == size(sharp)
end

@testset "make_synthetic_blur - with Gaussian noise" begin
    sharp = rand(Float64, 32, 32)
    B_noiseless, k = make_synthetic_blur(sharp; len=5)
    B_noisy, _ = make_synthetic_blur(sharp; len=5, noise_std=0.01)
    @test size(B_noiseless) == size(B_noisy) == size(sharp)
    @test std(B_noisy) >= std(B_noiseless) * 0.99  # noisy usually larger; allow rare tie
end

@testset "make_synthetic_blur - small image" begin
    sharp = rand(Float64, 16, 16)
    B, k = make_synthetic_blur(sharp; len=5)
    @test size(B) == size(sharp)
end

@testset "make_synthetic_blur - length 1 kernel" begin
    sharp = rand(Float64, 32, 32)
    B, k = make_synthetic_blur(sharp; len=1)
    @test size(B) == size(sharp)
    @test B ≈ sharp
end

@testset "make_synthetic_blur - preserves intensity range" begin
    sharp = rand(Float64, 32, 32) .* 0.5 .+ 0.25
    B, k = make_synthetic_blur(sharp; len=7)
    @test all(B .>= 0)
    @test all(B .<= 1)
end

@testset "deblur - basic grayscale" begin
    img = rand(Float64, 64, 64)
    B, k = make_synthetic_blur(img; len = 9)
    I, kest = deblur(B; kernel_size = 9)
    @test size(I) == size(img)
    @test sum(kest) ≈ 1.0
    @test all(isfinite, kest)
    @test all(isfinite, I)
end

@testset "deblur - with noise" begin
    img = rand(Float64, 48, 48)
    B, k = make_synthetic_blur(img; len = 7, noise_std=0.001)
    I, kest = deblur(B; kernel_size = 9)
    @test size(I) == size(img)
end

@testset "deblur - different kernel sizes" begin
    img = rand(Float64, 48, 48)
    for ks in [7, 11, 15]
        B, _ = make_synthetic_blur(img; len=ks-2)
        I, kest = deblur(B; kernel_size=ks)
        @test size(I) == size(img)
    end
end

@testset "estimate_kernel - basic" begin
    img = rand(Float64, 64, 64)
    B, k_true = make_synthetic_blur(img; len=9)
    k_est = estimate_kernel(B; kernel_size=9)
    @test size(k_est)[1] == 9
    @test size(k_est)[2] == 9
    @test sum(k_est) ≈ 1.0
    @test all(k_est .>= 0)
    @test all(isfinite, k_est)
end

@testset "deconvolve - with known kernel" begin
    img = rand(Float64, 48, 48)
    k = ones(5, 5) ./ 25
    B = imfilter(img, k)
    I = deconvolve(B, k)
    @test size(I) == size(img)
end

@testset "RobustMotionDeblur basic" begin
    img = rand(Float64, 64, 64)
    B, k = make_synthetic_blur(img; len = 9)
    I, kest = deblur(B; kernel_size = 9)
    @test size(I) == size(img)
end

@testset "estimate_kernel - synthetic ground-truth sanity" begin
    # Known horizontal motion kernel, small image, few levels/iters for speed
    k_true = motion_kernel(7; θ=0.0)
    sharp = rand(Float64, 48, 48)
    B, _ = make_synthetic_blur(sharp; len=7, θ=0.0)
    k_est = estimate_kernel(B; kernel_size=7, levels=2, inner_iters=2, max_iters=3)
    @test size(k_est) == size(k_true)
    @test sum(k_est) ≈ 1.0
    @test all(k_est .>= 0)
end

# Artificial test cases for every key aspect of the pipeline (solver, kernel estimation, color, edges, Phase 2)
include("test_pipeline_components.jl")

# SVG-inspired structured test images with quality assessment
include("test_svg_patterns.jl")
