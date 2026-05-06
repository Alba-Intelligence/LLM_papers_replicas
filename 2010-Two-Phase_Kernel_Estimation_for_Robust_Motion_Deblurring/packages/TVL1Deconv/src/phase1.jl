# Phase 1 kernel initialization (multi-scale, Eq. 6, 8)

import ..Utils: to_gray, grad_x, grad_y

"""
    downsample2(img, levels)

Build a pyramid by repeatedly downsampling by a factor of 2.
Returns a vector of images from coarsest to finest.
"""
function downsample2(img, levels::Integer)
    pyr = Vector{Array{Float64, 2}}(undef, levels)
    I = to_gray(img)
    for ℓ ∈ levels:-1:1
        pyr[ℓ] = I
        if ℓ > 1
            # simple 2× subsampling for the next coarser level
            I = I[1:2:end, 1:2:end]
        end
    end
    return pyr
end

"""
    fft_kernel_from_gradients(gx_s, gy_s, gx_B, gy_B; γ)

Closed-form kernel estimate following Eq. (6) in the frequency domain.
Returns a full-size kernel (same size as the image); callers typically
crop it to the desired support and renormalize.
"""
function fft_kernel_from_gradients(gx_s, gy_s, gx_B, gy_B; γ::Real)
    m, n = size(gx_s)
    FxIs = fft(gx_s)
    FyIs = fft(gy_s)
    FxB = fft(gx_B)
    FyB = fft(gy_B)

    num = conj.(FxIs) .* FxB .+ conj.(FyIs) .* FyB
    den = conj.(FxIs) .* FxIs .+ conj.(FyIs) .* FyIs .+ γ
    K = real.(ifft(num ./ den))
    return K
end

"""
    crop_kernel_center(K, ksize)

Crop a (ksize×ksize) or (kh×kw) kernel from array `K` and normalize it.
Applies fftshift then crops around the center-of-mass of the kernel (not the
geometric center), so the estimated kernel is not shifted left/right/up/down.
"""
function crop_kernel_center(K, ksize::Union{Int, Tuple{Int, Int}})
    h, w = size(K)
    kh, kw = ksize isa Int ? (ksize, ksize) : ksize

    # fftshift: move low-frequency / impulse content from (1,1) to the center
    Kshift = circshift(K, (h ÷ 2, w ÷ 2))
    # Use non-negative mass for center-of-mass so crop is centered on the actual kernel
    W = max.(Kshift, 0.0)
    sw = sum(W)
    if sw < 1e-12
        W = max.(abs.(Kshift), 0.0)
        sw = sum(W)
    end
    if sw >= 1e-12
        cy = sum(i * sum(W[i, :]) for i in 1:h) / sw
        cx = sum(j * sum(W[:, j]) for j in 1:w) / sw
        cy = clamp(cy, 1.0, Float64(h))
        cx = clamp(cx, 1.0, Float64(w))
    else
        cy = (h + 1) / 2
        cx = (w + 1) / 2
    end
    r_h = kh ÷ 2
    r_w = kw ÷ 2
    # Center crop on (cy, cx); ensure integer bounds stay in range
    ys = max(1, min(h - kh + 1, round(Int, cy) - r_h))
    ye = min(h, ys + kh - 1)
    ys = max(1, ye - kh + 1)
    xs = max(1, min(w - kw + 1, round(Int, cx) - r_w))
    xe = min(w, xs + kw - 1)
    xs = max(1, xe - kw + 1)
    k = Kshift[ys:ye, xs:xe]
    k .-= minimum(k)  # remove small negatives
    s = sum(k)
    s > 0 && (k ./= s)
    return k
end

"""
    coarse_image_from_kernel(B, k, gx_s, gy_s; λ)

Compute a coarse latent image using the spatial prior ‖∇I - ∇Is‖²
as in Eq. (8). Uses FFT-based closed form.
"""
function coarse_image_from_kernel(B, k, gx_s, gy_s; λ::Real)
    Bgray = to_gray(B)
    m, n = size(Bgray)

    # Pad kernel to image size (simple zero-padding, centered)
    Kpad = zeros(Float64, m, n)
    kh, kw = size(k)
    r1 = (m - kh) ÷ 2
    c1 = (n - kw) ÷ 2
    Kpad[(r1+1):(r1+kh), (c1+1):(c1+kw)] .= k

    # For convolution via FFT, move the centered kernel to the (1,1)
    # position (ifftshift) before taking the FFT.
    Kpad_shift = circshift(Kpad, (-m ÷ 2, -n ÷ 2))
    Fk = fft(Kpad_shift)
    FB = fft(Bgray)

    # Derivative operators (simple forward differences)
    Dx = zeros(Float64, m, n)
    Dy = zeros(Float64, m, n)
    Dx[1, 1:2] .= [-1.0, 1.0]
    Dy[1:2, 1] .= [-1.0, 1.0]
    Fx = fft(Dx)
    Fy = fft(Dy)

    FsIx = fft(gx_s)
    FsIy = fft(gy_s)

    # Standard Wiener filter: for min_I ||k⊗I - B||² + λ||∇I - ∇Is||², the
    # FFT solution is FI = (conj(Fk)·FB + λ·(conj(Fx)·FsIx + conj(Fy)·FsIy))
    #                      / (|Fk|² + λ·(|Fx|² + |Fy|²))
    num = conj.(Fk) .* FB .+ λ .* (conj.(Fx) .* FsIx .+ conj.(Fy) .* FsIy)
    den = Fk .* conj.(Fk) .+ λ .* (Fx .* conj.(Fx) .+ Fy .* conj.(Fy))

    I = real.(ifft(num ./ (den .+ eps(Float64))))
    return I
end

"""
    estimate_kernel_phase1(B; kernel_size=25, levels=3, inner_iters=5,
                           λ=2e-3, γ=1.0, τr_init=0.2, τs_init=1e-4,
                           decay_factor=1.1)

Phase 1: multi-scale kernel initialization (Algorithm 1).
Returns `(k0, Isx, Isy)` at the finest scale.

`kernel_size` can be an Int (square kernel) or a Tuple{Int,Int} (kh, kw).
"""
function estimate_kernel_phase1(
    B;
    kernel_size::Union{Int, Tuple{Int, Int}} = 25,
    levels::Int = 3,
    inner_iters::Int = 5,
    λ::Real = 2e-3,
    γ::Real = 1.0,
    τr_init::Real = 0.2,
    τs_init::Real = 1e-4,
    decay_factor::Real = 1.1,
)
    kh, kw = kernel_size isa Int ? (kernel_size, kernel_size) : kernel_size
    kh > 0 || throw(ArgumentError("kernel height must be > 0, got $kh"))
    kw > 0 || throw(ArgumentError("kernel width must be > 0, got $kw"))
    levels > 0 || throw(ArgumentError("levels must be > 0, got $levels"))
    inner_iters > 0 || throw(ArgumentError("inner_iters must be > 0, got $inner_iters"))
    decay_factor > 1 || throw(ArgumentError("decay_factor must be > 1, got $decay_factor"))

    if iseven(kh)
        kh += 1
        @warn "kernel height must be odd, using $(kh)" maxlog=1
    end
    if iseven(kw)
        kw += 1
        @warn "kernel width must be odd, using $(kw)" maxlog=1
    end

    ksize = (kh, kw)
    pyr = downsample2(B, levels)
    k = zeros(Float64, kh, kw)
    Isx_finest = nothing
    Isy_finest = nothing

    for ℓ ∈ 1:levels
        Bl = pyr[ℓ]
        τr = τr_init
        τs = τs_init
        gx_s = nothing
        gy_s = nothing
        # For alternating: use latent image for next iteration's edge selection (α blend with Bl)
        latent_blend_α = 0.3
        I_for_edges = Bl

        for _ ∈ 1:inner_iters
            Î = shock_filter(I_for_edges)
            gx_s, gy_s, _, _ = select_edges(Bl, Î, ksize; τr = τr, τs = τs)

            # Gradients of blurred image
            gxB = grad_x(Bl)
            gyB = grad_y(Bl)

            Kfull = fft_kernel_from_gradients(gx_s, gy_s, gxB, gyB; γ = γ)
            k = crop_kernel_center(Kfull, ksize)

            Il = coarse_image_from_kernel(Bl, k, gx_s, gy_s; λ = λ)
            # Next iteration: edges from blend of latent and blurred (paper alternating)
            I_for_edges = latent_blend_α .* Il .+ (1 - latent_blend_α) .* to_gray(Bl)

            τr /= decay_factor
            τs /= decay_factor
        end

        if ℓ == levels
            Isx_finest = gx_s
            Isy_finest = gy_s
        end
    end

    return k, Isx_finest, Isy_finest
end
