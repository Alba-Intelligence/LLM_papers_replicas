# Phase 2 ISD-based kernel refinement (Eq. 9-11)
# Paper: minimize E(k) = (1/2)‖∇I_s⊗k − ∇B‖² + γ Σ_{j∉S} |k_j| via IRLS (Eq. 10-11).
# No hard thresholding; sparsity via adaptive regularization on the complement of support S.

import ..Utils: to_gray, grad_x, grad_y
using LinearAlgebra

"""
    first_significant_jump(k, h, iter_idx; divisor=2)

Compute the threshold s for support detection (Eq. 9) using
the "first significant jump" rule from [22].
"""
function first_significant_jump(k::AbstractArray, h::Integer, iter_idx::Integer; divisor::Real=2)
    v = sort(vec(abs.(k)))
    n = length(v)
    if n <= 1
        return zero(eltype(k))
    end
    diffs = v[2:end] .- v[1:end-1]
    thresh = maximum(abs.(k)) / (divisor * h * max(iter_idx, 1))
    for j = 1:length(diffs)
        if diffs[j] > thresh
            return v[j]
        end
    end
    return v[end]
end

"""
    center_kernel(k)

Shift kernel `k` so that its center-of-mass is at the geometric center
of the array. This compensates for any global circular shift introduced
by FFT conventions and stabilises the deconvolution alignment.
"""
function center_kernel(k::AbstractMatrix)
    h, w = size(k)
    s = sum(k)
    # Degenerate or zero kernel: nothing to do
    abs(s) == 0 && return copy(k)

    # Center of mass in 1-based coordinates
    cy_num = zero(eltype(k))
    for i in 1:h
        cy_num += i * sum(k[i, :])
    end
    cy = cy_num / s

    cx_num = zero(eltype(k))
    for j in 1:w
        cx_num += j * sum(k[:, j])
    end
    cx = cx_num / s

    cy_target = (h + 1) / 2
    cx_target = (w + 1) / 2

    dy = round(Int, cy_target - cy)
    dx = round(Int, cx_target - cx)

    return circshift(k, (dy, dx))
end

# --- IRLS helpers for Eq. (10)-(11) ---
# A maps vec(k) to [vec(∂x I_s ⊗ k); vec(∂y I_s ⊗ k)]. So A^T A is (kh*kw)×(kh*kw).
# (A^T A)_{(a,b),(a',b')} = R_gx(a-a', b-b') + R_gy(a-a', b-b') with R = autocorrelation.

function _autocorr_center(G::AbstractMatrix, kh::Int, kw::Int)
    m, n = size(G)
    M, N = 2 * m - 1, 2 * n - 1
    g_pad = zeros(Float64, M, N)
    g_pad[1:m, 1:n] .= G
    circ = real.(ifft(conj.(fft(g_pad)) .* fft(g_pad)))
    # FFT convention: zero-lag at circ[1,1]; lag (di,dj) at circ[1+di, 1+dj] (with mod)
    # R[i,j] = correlation at (i-kh, j-kw) so R[kh,kw] = zero lag
    R = zeros(Float64, 2 * kh - 1, 2 * kw - 1)
    for i in 1:(2*kh-1), j in 1:(2*kw-1)
        di, dj = i - kh, j - kw
        R[i, j] = circ[mod1(1 + di, M), mod1(1 + dj, N)]
    end
    return R
end

function _crosscorr_valid(G::AbstractMatrix, H::AbstractMatrix, kh::Int, kw::Int)
    m, n = size(G)
    M, N = m + kh - 1, n + kw - 1
    g_pad = zeros(Float64, M, N)
    h_pad = zeros(Float64, M, N)
    g_pad[1:m, 1:n] .= G
    h_pad[1:m, 1:n] .= H
    xc = real.(ifft(conj.(fft(g_pad)) .* fft(h_pad)))
    return xc[1:kh, 1:kw]
end

function _build_ata_atb(gx_s, gy_s, gx_B, gy_B, kh, kw)
    R_gx = _autocorr_center(gx_s, kh, kw)
    R_gy = _autocorr_center(gy_s, kh, kw)
    ATA = zeros(Float64, kh * kw, kh * kw)
    for a in 1:kh, b in 1:kw
        p = (a - 1) * kw + b
        for a2 in 1:kh, b2 in 1:kw
            q = (a2 - 1) * kw + b2
            di, dj = a - a2 + kh, b - b2 + kw
            ATA[p, q] = R_gx[di, dj] + R_gy[di, dj]
        end
    end
    cx_gx = _crosscorr_valid(gx_s, gx_B, kh, kw)
    cx_gy = _crosscorr_valid(gy_s, gy_B, kh, kw)
    ATb = vec(cx_gx .+ cx_gy)
    return ATA, ATb
end

"""
    refine_kernel_isd(B, k0, Isx, Isy; γ=1.0, max_iters=5, tol=1e-3, divisor=2)

Phase 2: ISD-based kernel refinement (Algorithm 2, Eq. 9–11).
Iteratively: (1) detect support S = {j : k_j > s} with "first significant jump";
(2) minimize E(k) = (1/2)‖∇I_s⊗k − ∇B‖² + γ Σ_{j∉S} |k_j| via IRLS (Eq. 11).
No hard thresholding; sparsity is achieved by adaptive L1 regularization on S̄.
"""
function refine_kernel_isd(
    B,
    k0,
    Isx,
    Isy;
    γ::Real=1.0,
    max_iters::Int=5,
    tol::Real=1e-3,
    divisor::Real=2,
)
    max_iters > 0 || throw(ArgumentError("max_iters must be > 0, got $max_iters"))
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    divisor > 0 || throw(ArgumentError("divisor must be > 0, got $divisor"))

    if maximum(abs.(k0)) < 1e-6 || sum(k0 .> 1e-6) < 5
        return center_kernel(copy(k0))
    end

    Bgray = to_gray(B)
    gx_B = grad_x(Bgray)
    gy_B = grad_y(Bgray)
    kh, kw = size(k0)
    ATA, ATb = _build_ata_atb(Isx, Isy, gx_B, gy_B, kh, kw)

    k = copy(k0)
    k = k ./ max(sum(k), 1e-12)
    h = max(kh, kw)
    ε_Ψ = 1e-5

    for iter_idx in 1:max_iters
        s = first_significant_jump(k, h, iter_idx; divisor=divisor)
        # Support S = {j : k_j > s}; complement S̄ gets L1 penalty (Eq. 10)
        in_support = abs.(k) .> s
        # IRLS diagonal: D_j = 0 if j ∈ S, else γ/Ψ_j with Ψ_j = max(|k_j|, ε) (Eq. 11)
        Ψ = max.(abs.(k), ε_Ψ)
        w = ifelse.(in_support, 0.0, γ ./ Ψ)
        D = vec(w)
        # [A^T A + γ diag(1/Ψ for S̄)] V_k = A^T V_B  (Eq. 11)
        M = ATA + Diagonal(D)
        vk = M \ ATb
        knew = reshape(vk, kh, kw)
        # Non-negativity and normalize (kernel is PSF)
        knew = max.(knew, 0.0)
        ssum = sum(knew)
        if ssum > 0
            knew = knew ./ ssum
        else
            knew = copy(k)
        end
        rel = norm(knew - k) / (norm(k) + 1e-12)
        k = knew
        rel <= tol && break
    end

    return center_kernel(k)
end

"""
    estimate_kernel(B; kernel_size=25, levels=3, inner_iters=5, λ=2e-3, γ=1.0,
                    τr_init=0.2, τs_init=1e-4, decay_factor=1.1,
                    max_iters=5, tol=1e-3, divisor=2)

Convenience wrapper: Phase 1 initialization followed by Phase 2 refinement.
Returns `(ks, k0, Isx, Isy)`.

`kernel_size` can be an Int (square kernel) or a Tuple{Int,Int} (kh, kw).

# Arguments
- `kernel_size`: Size of the motion blur kernel
- `levels`: Number of scales in multi-scale kernel estimation (Phase 1)
- `inner_iters`: Number of inner iterations per scale (Phase 1)
- `λ`: Regularisation weight for the data term (Phase 1)
- `γ`: Weight in Phase 2 Eq. (10); paper uses γ=1 for refinement
- `τr_init`, `τs_init`, `decay_factor`: Phase 1 edge selection
- `max_iters`, `tol`, `divisor`: Phase 2 ISD (support detection + IRLS)
"""
function estimate_kernel(
    B;
    kernel_size::Union{Int, Tuple{Int,Int}} = 25,
    levels::Int = 3,
    inner_iters::Int = 5,
    λ::Real = 2e-3,
    γ::Real = 10.0,
    τr_init::Real = 0.2,
    τs_init::Real = 1e-4,
    decay_factor::Real = 1.1,
    max_iters::Int = 5,
    tol::Real = 1e-3,
    divisor::Real = 2,
)
    k0, Isx, Isy = estimate_kernel_phase1(
        B; kernel_size=kernel_size, levels=levels, inner_iters=inner_iters,
        λ=λ, γ=γ, τr_init=τr_init, τs_init=τs_init, decay_factor=decay_factor
    )
    ks = refine_kernel_isd(
        B, k0, Isx, Isy; γ=γ, max_iters=max_iters, tol=tol, divisor=divisor
    )
    return ks, k0, Isx, Isy
end
