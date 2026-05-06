using .Utils

# Default parameters for gradient confidence computation
const GRAD_CONF_EPS = 0.5  # Small constant in denominator (Eq. 2)
const SHOCK_FILTER_DT = 0.1  # Time step for shock filter
const SHOCK_FILTER_ITERS = 10  # Default iterations
const SHOCK_FILTER_SIGMA = 1.0  # Gaussian smoothing sigma

"""
    shock_filter(img; σ=1.0, dt=0.1, iters=10)

Discrete shock filtering (Osher–Rudin-type) used to sharpen salient step edges.
Implements Eq. (1) approximately:

∂I/∂t = -sign(ΔI) * ‖∇I‖,

starting from a Gaussian-smoothed version of the input.
"""
function shock_filter(img; σ::Real=SHOCK_FILTER_SIGMA, dt::Real=SHOCK_FILTER_DT, iters::Integer=SHOCK_FILTER_ITERS)
    σ > 0 || throw(ArgumentError("σ must be positive, got $σ"))
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    iters >= 1 || throw(ArgumentError("iters must be at least 1, got $iters"))

    I0 = Utils.to_gray(img)
    # Gaussian pre-smoothing
    smoothed = imfilter(I0, Kernel.gaussian((σ, σ)))

    I = copy(smoothed)

    for _ = 1:iters
        gx = grad_x(I)
        gy = grad_y(I)
        gradmag = sqrt.(gx .^ 2 .+ gy .^ 2 .+ eps(Float64))
        ΔI = laplacian(I)
        I .-= dt .* sign.(ΔI) .* gradmag
    end
    return I
end

"""
    gradient_confidence(B, h)

Compute the gradient confidence map r(x) as in Eq. (2) on a (grayscale) blurred image `B`,
using an h×h window (or max(hk, wk) if h is a tuple):

    r(x) = ‖∑_{y∈Nh(x)} ∇B(y)‖ / (∑_{y∈Nh(x)} ‖∇B(y)‖ + 0.5)
"""
function gradient_confidence(B, h::Union{Int, Tuple{Int,Int}})
    h isa Tuple && (h = max(h[1], h[2]))
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    isodd(h) || @warn "h should be odd for symmetric windows, got $h (using $h as-is)"

    Bgray = Utils.to_gray(B)
    gx = grad_x(Bgray)
    gy = grad_y(Bgray)
    m, n = size(Bgray)

    r = zeros(Float64, m, n)
    hw = h ÷ 2

    @inbounds for i in 1:m, j in 1:n
        sum_mag = 0.0
        sum_gx = 0.0
        sum_gy = 0.0
        count = 0
        for di in -hw:hw
            for dj in -hw:hw
                ii = clamp(i + di, 1, m)
                jj = clamp(j + dj, 1, n)
                g = sqrt(gx[ii, jj]^2 + gy[ii, jj]^2 + eps(Float64))
                sum_mag += g
                sum_gx += gx[ii, jj]
                sum_gy += gy[ii, jj]
                count += 1
            end
        end
        num = sqrt(sum_gx^2 + sum_gy^2)
        denom = sum_mag + GRAD_CONF_EPS
        r[i, j] = num / denom
    end
    return r
end

"""
    select_edges(B, Î_shock, h; τr, τs)

Compute the selective edge map ∇I_s used for kernel estimation, following
Eqs. (2)–(4). Returns `(gx_s, gy_s, r, M)` where:

- `gx_s`, `gy_s`: selected gradients of the shock-filtered image.
- `r`: gradient confidence map.
- `M`: binary mask H(r - τr).

`h` can be an Int (square window) or a Tuple{Int,Int} (uses max for window size).
"""
function select_edges(B, Î_shock, h::Union{Int, Tuple{Int,Int}}; τr::Real, τs::Real)
    τr >= 0 || throw(ArgumentError("τr must be non-negative, got $τr"))
    τs >= 0 || throw(ArgumentError("τs must be non-negative, got $τs"))

    # r map and mask M
    r = gradient_confidence(B, h)
    M = r .>= τr

    # Gradients of shock-filtered image
    gx = grad_x(Î_shock)
    gy = grad_y(Î_shock)
    g2 = gx .^ 2 .+ gy .^ 2

    # Heaviside-like selection: H(M * ‖∇Î‖² - τs)
    mask_edges = M .& (g2 .>= τs)

    gx_s = gx .* mask_edges
    gy_s = gy .* mask_edges

    return gx_s, gy_s, r, M
end
