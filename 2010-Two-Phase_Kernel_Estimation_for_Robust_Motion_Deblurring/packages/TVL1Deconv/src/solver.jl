# TV-L1 deconvolution solver (Eq. 12-18)
# Uses normaliseKernel, shrinkKernel, and to_gray from utils.jl
using .Utils

"""
    tvl1_deconv_gray(B, k; λ=2e-2, β0=1.0, θ0=1/λ, βmin=0.01, θmin=0.01,
                      max_outer=10, max_inner=5)

Solve the TV-L1 deconvolution problem for a single grayscale channel using
the half-quadratic splitting scheme (Eqs. 12–18, Algorithm 3).
"""
function tvl1_deconv_gray(
    B,
    k;
    λ::Real = 2e-2,
    β0::Real = 1.0,
    θ0::Real = 1 / λ,
    βmin::Real = 0.01,
    θmin::Real = 0.01,
    max_outer::Int = 10,
    max_inner::Int = 5,
)
    Bgray = Utils.to_gray(B)
    m, n = size(Bgray)

    # Flip kernel for proper deconvolution direction
    # k = rot180(k)

    # Pad kernel and precompute FFTs.
    # `pad_kernel` places `k` centered in `Kpad`; for convolution using FFT
    # we need the kernel origin at (1,1), so apply an "ifftshift"-like
    # circular shift before taking the FFT.

    # Pad PSF to image size
    psf = zeros(Float64, m, n)
    Kpad = k |> Utils.normaliseKernel |> Utils.shrinkKernel
    mₖ, nₖ = size(Kpad)

    # Place the kernel in the top left corner
    psf[1:mₖ, 1:nₖ] .= Kpad

    # Shift PSF so kernel origin at (1,1) for 2D convolution theorem
    psf_shift = circshift(psf, (-mₖ ÷ 2, -nₖ ÷ 2))
    Fk = fft(psf_shift)  # full 2D FFT for proper 2D convolution

    # Derivative operators in Fourier domain (simple forward differences)
    Dx = zeros(Float64, m, n)
    Dy = zeros(Float64, m, n)
    Dx[1, 1:2] .= [-1.0, 1.0]
    Dy[1:2, 1] .= [-1.0, 1.0]
    Fx = fft(Dx)
    Fy = fft(Dy)

    I = copy(Bgray)
    v = zeros(Float64, m, n)
    wx = zeros(Float64, m, n)
    wy = zeros(Float64, m, n)
    gx = zeros(Float64, m, n)
    gy = zeros(Float64, m, n)

    β = β0
    ϵ = eps(Float64)

    while β > βmin
        # Update v (Eq. 18) given current I
        FI = fft(I)
        Ik = real.(ifft(Fk .* FI))
        resid = Ik .- Bgray
        v = sign.(resid) .* max.(abs.(resid) .- β, 0.0)

        θ = θ0
        inner = 0
        while θ > θmin && inner < max_inner
            # Update w (Eq. 17) using finite differences
            fill!(gx, 0.0)
            fill!(gy, 0.0)
            @inbounds for i in 1:m, j in 1:n
                i2 = clamp(i+1, 1, m)
                j2 = clamp(j+1, 1, n)
                gx[i, j] = I[i, j2] - I[i, j]
                gy[i, j] = I[i2, j] - I[i, j]
            end
            gradmag = sqrt.(gx .^ 2 .+ gy .^ 2 .+ ϵ)
            shrink = max.(gradmag .- θ * λ, 0.0) ./ gradmag
            wx = gx .* shrink
            wy = gy .* shrink

            # Update I (Eq. 15): FI = (conj(Fk)·F(B+v) + (β/θ)·(conj(Fx)·Fwx + conj(Fy)·Fwy))
            #                        / (|Fk|² + (β/θ)·(|Fx|² + |Fy|²))
            FvB = fft(Bgray .+ v)
            Fwx = fft(wx)
            Fwy = fft(wy)

            num = conj.(Fk) .* FvB .+ (β / θ) .* (conj.(Fx) .* Fwx .+ conj.(Fy) .* Fwy)
            den = Fk .* conj.(Fk) .+ (β / θ) .* (Fx .* conj.(Fx) .+ Fy .* conj.(Fy))
            I = real.(ifft(num ./ (den .+ ϵ)))

            θ /= 2
            inner += 1
        end

        β /= 2
    end

    # Return spatial-domain image, clamped to [0,1]
    return clamp01nan.(I)
end

"""
    tvl1_deconv_joint(B_rgb, k; λ=2e-2, β0=1.0, θ0=1/λ, βmin=0.01, θmin=0.01,
                       max_outer=10, max_inner=5)

Solve the TV-L1 deconvolution problem for all three RGB channels jointly.

The joint optimization:
- Stacks all 3 channels into a 3D array
- Computes the gradient (TV term) from the combined luminance
- Updates all 3 channels together in each iteration
- Uses the same kernel for all channels
"""
function tvl1_deconv_joint(
    image_rgb,
    k;
    λ::Real = 2e-2,
    β0::Real = 1.0,
    θ0::Real = 1 / λ,
    βmin::Real = 0.01,
    θmin::Real = 0.01,
    max_outer::Int = 10,
    max_inner::Int = 5,
)
    B_rgb = float.(channelview(image_rgb))
    _, m, n = size(B_rgb)

    # Create zeros array same size as image, place kernel in top-left, shift, then 2D fft
    Kpad = zeros(Float64, m, n)
    Kpad_kernel = k |> Utils.normaliseKernel |> Utils.shrinkKernel
    mₖ, nₖ = size(Kpad_kernel)
    Kpad[1:mₖ, 1:nₖ] .= Kpad_kernel
    Fk = fft(circshift(Kpad, (-mₖ ÷ 2, -nₖ ÷ 2)))

    # Derivative operators in Fourier domain (simple forward differences)
    Dx = zeros(Float64, m, n)
    Dy = zeros(Float64, m, n)
    Dx[1, 1:2] .= [-1.0, 1.0]
    Dy[1:2, 1] .= [-1.0, 1.0]
    Fx = fft(Dx)
    Fy = fft(Dy)

    # Initialize I as 3-channel Float64 (avoid N0f8 clamping during iterations)
    I = copy(B_rgb)
    v = zeros(Float64, 3, m, n)
    wx = zeros(Float64, 3, m, n)
    wy = zeros(Float64, 3, m, n)

    β = β0
    ϵ = eps(Float64)

    # Luminance weights for TV computation
    lum_weights = [0.299, 0.587, 0.114]

    # Preallocate lum_I, gx, gy
    lum_I = zeros(Float64, m, n)
    gx = zeros(Float64, m, n)
    gy = zeros(Float64, m, n)


    while β > βmin
        # Update v (Eq. 18) for each channel given current I
        for c in 1:3
            FIc = fft(I[c, :, :])
            Ikc = real.(ifft(Fk .* FIc))
            resid = Ikc .- B_rgb[c, :, :]
            v[c, :, :] = real.(sign.(resid) .* max.(abs.(resid) .- β, 0.0))
        end

        θ = θ0
        inner = 0
        while θ > θmin && inner < max_inner
            # Compute luminance from current estimate for TV term
            fill!(lum_I, 0.0)

            for c in 1:3
                lum_I .+= lum_weights[c] .* I[c, :, :]
            end

            # Update w (Eq. 17) using finite differences on luminance
            fill!(gx, 0.0)
            fill!(gy, 0.0)

            @inbounds for i in 1:m, j in 1:n
                i2 = clamp(i+1, 1, m)
                j2 = clamp(j+1, 1, n)
                gx[i, j] = lum_I[i, j2] - lum_I[i, j]
                gy[i, j] = lum_I[i2, j] - lum_I[i, j]
            end

            gradmag = sqrt.(gx .^ 2 .+ gy .^ 2 .+ ϵ)
            shrink = max.(gradmag .- θ * λ, 0.0) ./ gradmag
            wx_shrink = gx .* shrink
            wy_shrink = gy .* shrink

            # Apply same shrinkage to all channels
            for c in 1:3
                wx[c, :, :] .= wx_shrink
                wy[c, :, :] .= wy_shrink
            end

            # Update I (Eq. 15) for all channels together
            for c in 1:3
                FvBc = fft(B_rgb[c, :, :] .+ v[c, :, :])
                Fwxc = fft(wx[c, :, :])
                Fwyc = fft(wy[c, :, :])

                num = conj.(Fk) .* FvBc .+ (β / θ) .* (conj.(Fx) .* Fwxc .+ conj.(Fy) .* Fwyc)
                den = Fk .* conj.(Fk) .+ (β / θ) .* (Fx .* conj.(Fx) .+ Fy .* conj.(Fy))
                I[c, :, :] = clamp01nan.(real.(ifft(num ./ (den .+ ϵ))))
            end

            θ /= 2
            inner += 1
        end

        β /= 2
    end

    # return I, clamp01nan.(irfft(Fk, m))
    return I
end

"""
    tvl1_deconvolve(B, k; λ=2e-2, kwargs...)

High-level TV-L1 deconvolution entry point (Algorithm 3).

If `B` is color, the channels are processed jointly using a single optimization
loop that computes the TV term from luminance and updates all channels together.

If `B` is grayscale, processes the single channel directly.

The kernel is normalized (to sum to 1 with positive values) and shrunk
(removing zero rows/columns, ensuring odd dimensions) at entry.
"""
function tvl1_deconvolve(B, k; λ::Real = 2e-2, kwargs...)
    # Normalize and shrink kernel at entry point
    k = normaliseKernel(k)
    k = shrinkKernel(k)
    mₖ, nₖ = size(k)

    # Replicate-pad to reduce FFT boundary ripples; then crop result to original size
    pad_y, pad_x = mₖ, nₖ

    if ndims(channelview(B)) == 2
        m, n = size(B)
        B_pad_raw = ImageFiltering.padarray(Utils.to_gray(B), ImageFiltering.Pad(:replicate, pad_y, pad_x))
        s1, s2 = size(B_pad_raw)
        B_pad = Array{Float64}(undef, s1, s2)
        for (ii, i) in enumerate(axes(B_pad_raw, 1)), (jj, j) in enumerate(axes(B_pad_raw, 2))
            B_pad[ii, jj] = B_pad_raw[i, j]
        end
        result = tvl1_deconv_gray(B_pad, k; λ = λ, kwargs...)
        return result[pad_y+1:pad_y+m, pad_x+1:pad_x+n]
    end

    # Color image: use joint optimization (copy to 1-based so no OffsetArray in solver)
    B_cv = float.(channelview(B))
    _, m, n = size(B_cv)
    B_cv_pad_raw = ImageFiltering.padarray(B_cv, ImageFiltering.Pad(:replicate, 0, pad_y, pad_x))
    s1, s2, s3 = size(B_cv_pad_raw)
    B_cv_pad = Array{eltype(B_cv)}(undef, s1, s2, s3)
    for (ii, i) in enumerate(axes(B_cv_pad_raw, 1)), (jj, j) in enumerate(axes(B_cv_pad_raw, 2)), (kk, k) in enumerate(axes(B_cv_pad_raw, 3))
        B_cv_pad[ii, jj, kk] = B_cv_pad_raw[i, j, k]
    end
    B_pad = Images.colorview(Images.RGB, B_cv_pad)
    result = tvl1_deconv_joint(B_pad, k; λ = λ, kwargs...)
    return result[:, pad_y+1:pad_y+m, pad_x+1:pad_x+n]
end
