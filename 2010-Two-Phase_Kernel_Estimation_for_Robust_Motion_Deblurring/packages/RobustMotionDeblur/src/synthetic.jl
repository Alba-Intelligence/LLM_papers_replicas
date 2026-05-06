# Synthetic blur utilities

"""
    motion_kernel(len; θ=0.0)

Create a simple linear motion blur kernel of length `len` pixels
at angle `θ` (in radians), normalized to sum to one.
"""
function motion_kernel(len::Integer; θ::Real=0.0)
    len = max(len, 1)
    # Draw a line in a small support and normalize
    r = len ÷ 2
    sizek = 2r + 1
    k = zeros(Float64, sizek, sizek)
    cx = cy = r + 1
    for i in 1:sizek, j in 1:sizek
        x = j - cx
        y = i - cy
        # project onto direction (cosθ, sinθ)
        t = x * cos(θ) + y * sin(θ)
        if abs(t) < 0.5 && abs(-x * sin(θ) + y * cos(θ)) <= r + 0.5
            k[i, j] = 1.0
        end
    end
    s = sum(k)
    s > 0 && (k ./= s)
    return k
end

"""
    curved_motion_kernel(len; curvature=0.3, θ=0.0)

Create an arc-shaped motion blur kernel. The path is a circular arc of
`len` pixels starting at angle `θ`, bending by `curvature` radians per
pixel (positive = counter-clockwise).

Returns a normalized kernel whose support is `(2r+1) × (2r+1)` where
`r = len ÷ 2`.
"""
function curved_motion_kernel(len::Integer; curvature::Real=0.3, θ::Real=0.0)
    len = max(len, 1)
    r = len ÷ 2
    sizek = 2r + 1
    k = zeros(Float64, sizek, sizek)
    cx = cy = r + 1

    # Walk along an arc: at each step advance 1 pixel in the current direction,
    # then rotate the direction by `curvature`.
    nsteps = max(len * 4, 100)  # oversample for smoother anti-aliasing
    x, y = 0.0, 0.0
    angle = θ
    dt = Float64(len - 1) / nsteps

    for t in 0:nsteps
        # Rasterise the current position with bilinear splatting
        fi, fj = cy + y, cx + x
        i0 = clamp(floor(Int, fi), 1, sizek)
        j0 = clamp(floor(Int, fj), 1, sizek)
        i1 = clamp(i0 + 1, 1, sizek)
        j1 = clamp(j0 + 1, 1, sizek)
        fy = fi - i0
        fx = fj - j0
        k[i0, j0] += (1 - fy) * (1 - fx)
        k[i0, j1] += (1 - fy) * fx
        k[i1, j0] += fy * (1 - fx)
        k[i1, j1] += fy * fx

        x += dt * cos(angle)
        y += dt * sin(angle)
        angle += curvature * dt
    end

    s = sum(k)
    s > 0 && (k ./= s)
    return k
end

"""
    random_walk_kernel(len; seed=nothing)

Create a random-walk motion blur kernel of approximately `len` pixels.
The walk takes small random steps, producing a realistic non-linear blur
trajectory. Set `seed` for reproducible results.

Returns a normalized kernel whose support is `(2r+1) × (2r+1)` where
`r = len ÷ 2`.
"""
function random_walk_kernel(len::Integer; seed::Union{Int,Nothing}=nothing)
    len = max(len, 1)
    r = len ÷ 2
    sizek = 2r + 1
    k = zeros(Float64, sizek, sizek)
    cx = cy = r + 1

    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    nsteps = max(len * 4, 100)
    x, y = 0.0, 0.0
    # Random smooth trajectory via accumulated random direction changes
    angle = 2π * rand(rng)
    step_size = Float64(len - 1) / nsteps

    for _ in 0:nsteps
        fi, fj = cy + y, cx + x
        i0 = clamp(floor(Int, fi), 1, sizek)
        j0 = clamp(floor(Int, fj), 1, sizek)
        i1 = clamp(i0 + 1, 1, sizek)
        j1 = clamp(j0 + 1, 1, sizek)
        fy = fi - i0
        fx = fj - j0
        k[i0, j0] += (1 - fy) * (1 - fx)
        k[i0, j1] += (1 - fy) * fx
        k[i1, j0] += fy * (1 - fx)
        k[i1, j1] += fy * fx

        # Brownian-like direction change
        angle += 0.5 * randn(rng)
        x += step_size * cos(angle)
        y += step_size * sin(angle)
        # Keep within kernel support
        x = clamp(x, -r, r)
        y = clamp(y, -r, r)
    end

    s = sum(k)
    s > 0 && (k ./= s)
    return k
end

"""
    make_synthetic_blur(I; len=15, θ=0.0, noise_std=0.0, impulse_prob=0.0, kernel=nothing)

Generate a synthetic blurred image from a sharp input `I` by convolving
with a motion kernel of given length/angle and optionally adding Gaussian
and impulse noise. Returns `(B, k)`.

If `kernel` is provided, it is used directly instead of generating a
linear motion kernel from `len` and `θ`.
"""
function make_synthetic_blur(
    I;
    len::Int=15,
    θ::Real=0.0,
    noise_std::Real=0.0,
    impulse_prob::Real=0.0,
    kernel::Union{Nothing,AbstractMatrix}=nothing,
)
    k = isnothing(kernel) ? motion_kernel(len; θ=θ) : kernel
    B = imfilter(I, k)

    if noise_std > 0
        B .= clamp.(B .+ noise_std .* randn(eltype(B), size(B)), zero(eltype(B)), one(eltype(B)))
    end

    if impulse_prob > 0
        mask = rand(size(B)) .< impulse_prob
        # Salt-and-pepper: randomly set to 0 or 1
        salt = rand(sum(mask)) .> 0.5
        B[mask] .= 0
        B[mask][salt] .= 1
    end

    return B, k
end

