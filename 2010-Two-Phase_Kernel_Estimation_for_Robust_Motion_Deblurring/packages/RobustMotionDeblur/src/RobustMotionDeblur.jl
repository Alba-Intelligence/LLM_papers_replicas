module RobustMotionDeblur

using Images, ImageIO, FileIO, Downloads, Statistics, Random
using TVL1Deconv

include("synthetic.jl")
include("data.jl")
include("visualize.jl")

"""
    edgetaper(B; margin=16)

Apply a simple cosine edge taper to image `B` to reduce FFT ringing.
Pixels near the border are blended towards the image mean.
"""
function edgetaper(B; margin::Int=16)
    ndims(B) == 2 || ndims(B) == 3 || error("Unsupported image dimensionality")

    if ndims(B) == 2
        m, n = size(B)
    else
        _, m, n = size(channelview(B))
    end

    margin = min(margin, min(m, n) ÷ 2)
    if margin <= 0
        return B
    end

    wy = ones(Float64, m)
    wx = ones(Float64, n)
    for i in 1:margin
        t = 0.5 * (1 - cos(pi * (i - 1) / margin))
        wy[i] = min(wy[i], t)
        wy[end-i+1] = min(wy[end-i+1], t)
        wx[i] = min(wx[i], t)
        wx[end-i+1] = min(wx[end-i+1], t)
    end
    Wy = max.(1 .- wy, 0)  # 1 in centre, 0 at extreme border
    Wx = max.(1 .- wx, 0)
    w = Wy .* Wx'

    μ = mean(B)
    C = similar(B)
    @. C = w * (B - μ) + μ
    return C
end

export deblur, estimate_kernel, deconvolve, make_synthetic_blur, motion_kernel
export curved_motion_kernel, random_walk_kernel

"""High-level blind deblurring: estimate kernel then deconvolve."""
function deblur(
    B;
    kernel_size::Union{Int,Tuple{Int,Int}}=25,
    # Phase 1 parameters
    levels::Int=3,
    inner_iters::Int=5,
    λ::Real=2e-3,
    γ::Real=1.0,
    τr_init::Real=0.2,
    τs_init::Real=1e-4,
    decay_factor::Real=1.1,
    # Phase 2 parameters
    max_iters::Int=5,
    tol::Real=1e-3,
    divisor::Real=2,
    # TV-L1 deconvolution parameters (nothing → use scale-aware defaults from tvl1_defaults)
    max_outer::Union{Int,Nothing}=nothing,
    max_inner::Union{Int,Nothing}=nothing,
)
    Bt = edgetaper(B)
    k̂ = estimate_kernel(
        Bt; kernel_size=kernel_size,
        levels=levels, inner_iters=inner_iters,
        λ=λ, γ=γ, τr_init=τr_init, τs_init=τs_init, decay_factor=decay_factor,
        max_iters=max_iters, tol=tol, divisor=divisor,
    )
    I = deconvolve(Bt, k̂; λ=λ, max_outer=max_outer, max_inner=max_inner)
    return I, k̂
end

"""Estimate blur kernel only (wrapper around KernelEstimation)."""
function estimate_kernel(
    B;
    kernel_size::Union{Int,Tuple{Int,Int}}=25,
    # Phase 1 parameters
    levels::Int=3,
    inner_iters::Int=5,
    λ::Real=2e-3,
    γ::Real=1.0,
    τr_init::Real=0.2,
    τs_init::Real=1e-4,
    decay_factor::Real=1.1,
    # Phase 2 parameters
    max_iters::Int=5,
    tol::Real=1e-3,
    divisor::Real=2,
)
    ks, k0, Isx, Isy = TVL1Deconv.estimate_kernel(
        B; kernel_size=kernel_size,
        levels=levels, inner_iters=inner_iters,
        λ=λ, γ=γ, τr_init=τr_init, τs_init=τs_init, decay_factor=decay_factor,
        max_iters=max_iters, tol=tol, divisor=divisor,
    )
    return ks
end

"""Heuristic scale-aware defaults for TV-L1 parameters."""
function tvl1_defaults(B)
    A = float.(channelview(B))
    if ndims(A) == 3
        _, m, n = size(A)
    else
        m, n = size(A)
    end
    s = max(m, n)
    # Larger images: stronger data term and fewer outer iterations.
    if s <= 128
        return (λ=2e-2, max_outer=10, max_inner=5)
    elseif s <= 512
        return (λ=2.5e-2, max_outer=8, max_inner=4)
    else
        return (λ=4e-2, max_outer=6, max_inner=3)
    end
end

"""Non-blind deconvolution given known kernel."""
function deconvolve(B, k; λ::Real=2e-2, max_outer::Union{Int,Nothing}=nothing, max_inner::Union{Int,Nothing}=nothing, kwargs...)
    defs = tvl1_defaults(B)
    mo = isnothing(max_outer) ? defs.max_outer : max_outer
    mi = isnothing(max_inner) ? defs.max_inner : max_inner
    return TVL1Deconv.tvl1_deconvolve(B, k; λ=λ, max_outer=mo, max_inner=mi, kwargs...)
end

end # module
