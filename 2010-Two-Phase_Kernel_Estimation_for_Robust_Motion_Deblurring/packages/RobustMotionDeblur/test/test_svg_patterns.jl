# SVG-inspired test images for deblurring quality assessment.
# Each generator draws a structured pattern described by its SVG equivalent,
# then blurs with a known kernel and checks non-blind / blind recovery.
#
# Included from runtests.jl; requires Test, Images, ImageFiltering, Statistics,
# RobustMotionDeblur, TVL1Deconv already loaded (plus helpers from
# test_pipeline_components.jl which is included first).

# ═══════════════════════════════════════════════════════════════════════════
# SVG pattern generators
# ═══════════════════════════════════════════════════════════════════════════

"""
    svg_crosshair(n; stroke=3, bg=0.1, fg=0.9)

SVG equivalent:
```xml
<svg width="n" height="n">
  <rect width="100%" height="100%" fill="bg"/>
  <line x1="0" y1="50%" x2="100%" y2="50%" stroke="fg" stroke-width="stroke"/>
  <line x1="50%" y1="0" x2="50%" y2="100%" stroke="fg" stroke-width="stroke"/>
</svg>
```
Two perpendicular lines through the centre — exercises edge recovery in
both horizontal and vertical directions simultaneously.
"""
function svg_crosshair(n::Int; stroke::Int=3, bg=0.1, fg=0.9)
    img = fill(Float64(bg), n, n)
    cx, cy = n ÷ 2, n ÷ 2
    half = stroke ÷ 2
    # Horizontal bar
    img[max(1,cy-half):min(n,cy+half), :] .= fg
    # Vertical bar
    img[:, max(1,cx-half):min(n,cx+half)] .= fg
    return img
end

"""
    svg_circle(n; r_frac=0.35, stroke=2, bg=0.1, fg=0.9)

SVG equivalent:
```xml
<svg width="n" height="n">
  <rect width="100%" height="100%" fill="bg"/>
  <circle cx="50%" cy="50%" r="r_frac*n/2" stroke="fg"
          stroke-width="stroke" fill="none"/>
</svg>
```
Ring with edges at every orientation — a strong test for isotropic
deblurring fidelity.
"""
function svg_circle(n::Int; r_frac=0.35, stroke::Int=2, bg=0.1, fg=0.9)
    img = fill(Float64(bg), n, n)
    cx = cy = (n + 1) / 2
    R = r_frac * n / 2
    half = stroke / 2
    for i in 1:n, j in 1:n
        d = sqrt((i - cy)^2 + (j - cx)^2)
        if abs(d - R) <= half
            img[i, j] = fg
        end
    end
    return img
end

"""
    svg_checkerboard(n; cells=8, lo=0.15, hi=0.85)

SVG equivalent (conceptual):
```xml
<svg width="n" height="n">
  <!-- cells × cells checkerboard -->
  <rect ... fill="lo" or "hi" alternating />
</svg>
```
Rich in edges in both directions — ideal for testing kernel estimation
on multi-directional structure.
"""
function svg_checkerboard(n::Int; cells::Int=8, lo=0.15, hi=0.85)
    img = fill(Float64(lo), n, n)
    cw = n ÷ cells
    for i in 1:n, j in 1:n
        bi = div(i - 1, max(cw, 1))
        bj = div(j - 1, max(cw, 1))
        img[i, j] = ((bi + bj) % 2 == 0) ? lo : hi
    end
    return img
end

"""
    svg_starburst(n; rays=12, stroke=2, bg=0.1, fg=0.9)

SVG equivalent:
```xml
<svg width="n" height="n">
  <rect width="100%" height="100%" fill="bg"/>
  <!-- `rays` lines from centre to edge at equally-spaced angles -->
  <line x1="50%" y1="50%" x2="..." y2="..." stroke="fg" .../>
</svg>
```
Lines emanating from the centre at many angles — tests whether the
solver preserves directional detail uniformly.
"""
function svg_starburst(n::Int; rays::Int=12, stroke::Int=2, bg=0.1, fg=0.9)
    img = fill(Float64(bg), n, n)
    cx = cy = (n + 1) / 2
    half = stroke / 2
    R = n / 2 - 2
    for k in 0:rays-1
        θ = 2π * k / rays
        for t in 0.0:0.3:R
            x = cx + t * cos(θ)
            y = cy + t * sin(θ)
            i0 = round(Int, y)
            j0 = round(Int, x)
            for di in -ceil(Int, half):ceil(Int, half)
                for dj in -ceil(Int, half):ceil(Int, half)
                    ii, jj = i0 + di, j0 + dj
                    if 1 <= ii <= n && 1 <= jj <= n
                        # perpendicular distance to the ray
                        dx = jj - cx; dy = ii - cy
                        perp = abs(-sin(θ) * dx + cos(θ) * dy)
                        perp <= half && (img[ii, jj] = fg)
                    end
                end
            end
        end
    end
    return img
end

"""
    svg_nested_rects(n; rects=4, bg=0.1, fg=0.9)

SVG equivalent:
```xml
<svg width="n" height="n">
  <rect width="100%" height="100%" fill="bg"/>
  <rect x="m" y="m" width="n-2m" height="n-2m" fill="none"
        stroke="fg" stroke-width="2"/>   <!-- repeated at different margins -->
</svg>
```
Concentric rectangles — tests axis-aligned edge recovery at multiple
scales / positions.
"""
function svg_nested_rects(n::Int; rects::Int=4, bg=0.1, fg=0.9)
    img = fill(Float64(bg), n, n)
    for k in 1:rects
        margin = round(Int, n * k / (2rects + 2))
        i1, i2 = margin + 1, n - margin
        j1, j2 = margin + 1, n - margin
        (i1 > i2 || j1 > j2) && continue
        img[i1:min(i1+1,i2), j1:j2] .= fg   # top edge
        img[max(i2-1,i1):i2, j1:j2] .= fg   # bottom edge
        img[i1:i2, j1:min(j1+1,j2)] .= fg   # left edge
        img[i1:i2, max(j2-1,j1):j2] .= fg   # right edge
    end
    return img
end

"""
    svg_diagonal_cross(n; stroke=3, bg=0.1, fg=0.9)

SVG equivalent:
```xml
<svg width="n" height="n">
  <rect width="100%" height="100%" fill="bg"/>
  <line x1="0" y1="0" x2="100%" y2="100%" stroke="fg" stroke-width="stroke"/>
  <line x1="100%" y1="0" x2="0" y2="100%" stroke="fg" stroke-width="stroke"/>
</svg>
```
X-shaped cross on a diagonal — exercises 45° edge recovery, which is
harder than axis-aligned for FFT-based solvers.
"""
function svg_diagonal_cross(n::Int; stroke::Int=3, bg=0.1, fg=0.9)
    img = fill(Float64(bg), n, n)
    half = stroke / 2
    for i in 1:n, j in 1:n
        d1 = abs(i - j) / √2            # distance from main diagonal
        d2 = abs(i - (n + 1 - j)) / √2  # distance from anti-diagonal
        (d1 <= half || d2 <= half) && (img[i, j] = fg)
    end
    return img
end

"""
    svg_gradient_disc(n; bg=0.0)

SVG equivalent (conceptual):
```xml
<svg width="n" height="n">
  <radialGradient id="g" cx="50%" cy="50%" r="45%">
    <stop offset="0%" stop-color="white"/>
    <stop offset="100%" stop-color="black"/>
  </radialGradient>
  <circle cx="50%" cy="50%" r="45%" fill="url(#g)"/>
</svg>
```
Smooth radial gradient — no hard edges. Tests whether the solver
preserves smooth structure without hallucinating ringing artefacts.
"""
function svg_gradient_disc(n::Int; bg=0.0)
    img = fill(Float64(bg), n, n)
    cx = cy = (n + 1) / 2
    R = 0.45 * n
    for i in 1:n, j in 1:n
        d = sqrt((i - cy)^2 + (j - cx)^2)
        if d <= R
            img[i, j] = 1.0 - d / R
        end
    end
    return img
end

"""
    svg_letter_E(n; stroke=3, bg=0.1, fg=0.9)

SVG equivalent (conceptual):
```xml
<svg width="n" height="n">
  <rect width="100%" height="100%" fill="bg"/>
  <!-- Capital E drawn with horizontal + vertical strokes -->
</svg>
```
Glyph-like structure with mixed horizontal and vertical strokes —
realistic proxy for text deblurring.
"""
function svg_letter_E(n::Int; stroke::Int=3, bg=0.1, fg=0.9)
    img = fill(Float64(bg), n, n)
    # Proportional glyph occupying ~60% of the canvas
    m = round(Int, 0.2 * n)
    t = max(stroke, 2)
    x1, x2 = m, n - m         # left / right extent
    y1, y2 = m, n - m         # top / bottom extent
    ymid = (y1 + y2) ÷ 2
    # Vertical spine
    img[y1:y2, x1:min(x1+t-1, n)] .= fg
    # Top bar
    img[y1:min(y1+t-1, n), x1:x2] .= fg
    # Middle bar (slightly shorter)
    img[ymid:min(ymid+t-1, n), x1:x2-round(Int, 0.15n)] .= fg
    # Bottom bar
    img[max(y2-t+1, 1):y2, x1:x2] .= fg
    return img
end

# ═══════════════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════════════

const _SVG_FIG_DIR = joinpath(@__DIR__, "figures", "svg")

# ── Helper: run a non-blind deconv test on a pattern ─────────────────────

function _nonblind_test(name, sharp, k; λ=0.004, max_outer=12, max_inner=3,
                        mse_thresh=0.06, ssim_thresh=0.3)
    B, _ = make_synthetic_blur(sharp; kernel=k)
    B = Float64.(B)
    dec = deconvolve(B, k; λ=λ, max_outer=max_outer, max_inner=max_inner)
    dec = Float64.(replace(dec, NaN => 0.0))

    @test size(dec) == size(sharp)
    @test all(isfinite, dec)

    err      = mse(dec, sharp)
    ssim_val = ssim(dec, sharp)
    psnr_val = psnr(err)

    @test err < mse_thresh
    @test ssim_val > ssim_thresh

    RobustMotionDeblur.save_composite_summary(
        sharp, B, dec, k, k,
        joinpath(_SVG_FIG_DIR, "$(name)_nonblind.png"))

    println("   [$name non-blind] MSE=$(round(err,digits=5)) PSNR=$(round(psnr_val,digits=1))dB SSIM=$(round(ssim_val,digits=3))")
    return (mse=err, psnr=psnr_val, ssim=ssim_val)
end

# ── Helper: run a blind deblur test on a pattern ─────────────────────────

function _blind_test(name, sharp, k; kernel_size=nothing, levels=3,
                     mse_thresh=0.12, ssim_thresh=0.15, ncc_thresh=0.05)
    B, _ = make_synthetic_blur(sharp; kernel=k)
    B = Float64.(B)
    ks = isnothing(kernel_size) ? size(k, 1) : kernel_size
    I_deblur, k_est = deblur(B; kernel_size=ks, levels=levels)
    I_deblur = Float64.(replace(I_deblur, NaN => 0.0))

    @test size(I_deblur) == size(sharp)
    @test all(isfinite, I_deblur)
    @test sum(k_est) ≈ 1.0 atol=0.15

    err      = mse(I_deblur, sharp)
    ssim_val = ssim(I_deblur, sharp)
    ncc_k    = ncc(k_est, k)

    # Blind deblurring is harder — thresholds are relaxed
    @test err < mse_thresh
    @test ssim_val > ssim_thresh
    @test abs(ncc_k) > ncc_thresh

    RobustMotionDeblur.save_composite_summary(
        sharp, B, I_deblur, k, k_est,
        joinpath(_SVG_FIG_DIR, "$(name)_blind.png"))

    println("   [$name blind] MSE=$(round(err,digits=5)) SSIM=$(round(ssim_val,digits=3)) kernel_NCC=$(round(ncc_k,digits=3))")
    return (mse=err, ssim=ssim_val, ncc_k=ncc_k)
end

# ═══════════════════════════════════════════════════════════════════════════
# 1. Crosshair  — horizontal + vertical edges
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Crosshair: non-blind with horizontal kernel" begin
    sharp = svg_crosshair(128)
    k = motion_kernel(11; θ=0.0)
    _nonblind_test("crosshair_horiz", sharp, k; ssim_thresh=0.4)
end

@testset "SVG Crosshair: non-blind with diagonal kernel" begin
    sharp = svg_crosshair(128)
    k = motion_kernel(11; θ=π/4)
    _nonblind_test("crosshair_diag", sharp, k; ssim_thresh=0.35)
end

@testset "SVG Crosshair: blind with horizontal kernel" begin
    sharp = svg_crosshair(128)
    k = motion_kernel(9; θ=0.0)
    _blind_test("crosshair_horiz", sharp, k; ssim_thresh=0.01, mse_thresh=0.15)
end

# ═══════════════════════════════════════════════════════════════════════════
# 2. Circle  — edges at all orientations
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Circle: non-blind with horizontal kernel" begin
    sharp = svg_circle(128; r_frac=0.35, stroke=3)
    k = motion_kernel(11; θ=0.0)
    _nonblind_test("circle_horiz", sharp, k; ssim_thresh=0.35)
end

@testset "SVG Circle: non-blind with curved kernel" begin
    sharp = svg_circle(128; r_frac=0.35, stroke=3)
    k = curved_motion_kernel(11; θ=0.0, curvature=0.3)
    _nonblind_test("circle_curved", sharp, k; ssim_thresh=0.3)
end

# ═══════════════════════════════════════════════════════════════════════════
# 3. Checkerboard  — rich in both H & V edges
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Checkerboard: non-blind with horizontal kernel" begin
    sharp = svg_checkerboard(128; cells=8)
    k = motion_kernel(9; θ=0.0)
    _nonblind_test("checker_horiz", sharp, k; mse_thresh=0.12, ssim_thresh=0.3)
end

@testset "SVG Checkerboard: non-blind with vertical kernel" begin
    sharp = svg_checkerboard(128; cells=8)
    k = motion_kernel(9; θ=π/2)
    _nonblind_test("checker_vert", sharp, k; mse_thresh=0.12, ssim_thresh=0.3)
end

@testset "SVG Checkerboard: blind with horizontal kernel" begin
    sharp = svg_checkerboard(128; cells=8)
    k = motion_kernel(9; θ=0.0)
    _blind_test("checker_horiz", sharp, k; ssim_thresh=0.01, mse_thresh=0.15)
end

# ═══════════════════════════════════════════════════════════════════════════
# 4. Starburst  — multi-directional radiating lines
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Starburst: non-blind with horizontal kernel" begin
    sharp = svg_starburst(128; rays=12, stroke=2)
    k = motion_kernel(11; θ=0.0)
    _nonblind_test("starburst_horiz", sharp, k; ssim_thresh=0.3)
end

@testset "SVG Starburst: non-blind with random-walk kernel" begin
    sharp = svg_starburst(128; rays=12, stroke=2)
    k = random_walk_kernel(11; seed=7)
    _nonblind_test("starburst_rw", sharp, k; ssim_thresh=0.25)
end

# ═══════════════════════════════════════════════════════════════════════════
# 5. Nested rectangles  — axis-aligned edges at multiple scales
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Nested rects: non-blind with horizontal kernel" begin
    sharp = svg_nested_rects(128; rects=4)
    k = motion_kernel(11; θ=0.0)
    _nonblind_test("nestrects_horiz", sharp, k; ssim_thresh=0.35)
end

@testset "SVG Nested rects: blind with vertical kernel" begin
    sharp = svg_nested_rects(128; rects=4)
    k = motion_kernel(9; θ=π/2)
    _blind_test("nestrects_vert", sharp, k; ssim_thresh=0.01, mse_thresh=0.15)
end

# ═══════════════════════════════════════════════════════════════════════════
# 6. Diagonal cross (X)  — 45° edges
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Diagonal cross: non-blind with horizontal kernel" begin
    sharp = svg_diagonal_cross(128; stroke=3)
    k = motion_kernel(11; θ=0.0)
    _nonblind_test("diagcross_horiz", sharp, k; ssim_thresh=0.35)
end

@testset "SVG Diagonal cross: non-blind with curved kernel" begin
    sharp = svg_diagonal_cross(128; stroke=3)
    k = curved_motion_kernel(11; θ=π/6, curvature=0.25)
    _nonblind_test("diagcross_curved", sharp, k; ssim_thresh=0.3)
end

# ═══════════════════════════════════════════════════════════════════════════
# 7. Gradient disc  — smooth structure, no hard edges
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Gradient disc: non-blind with horizontal kernel" begin
    sharp = svg_gradient_disc(128)
    k = motion_kernel(11; θ=0.0)
    _nonblind_test("graddisc_horiz", sharp, k; ssim_thresh=0.5)
end

@testset "SVG Gradient disc: non-blind preserves smoothness" begin
    sharp = svg_gradient_disc(128)
    k = motion_kernel(9; θ=π/3)
    B, _ = make_synthetic_blur(sharp; kernel=k)
    B = Float64.(B)
    dec = deconvolve(B, k; λ=0.008, max_outer=10, max_inner=3)
    dec = Float64.(replace(dec, NaN => 0.0))
    # Laplacian energy: deblurred should not be much noisier than original
    Δ = [0 1 0; 1 -4 1; 0 1 0]
    lap_sharp = imfilter(sharp, Δ)
    lap_dec   = imfilter(dec, Δ)
    energy_ratio = sum(lap_dec .^ 2) / max(sum(lap_sharp .^ 2), 1e-10)
    @test energy_ratio < 5.0   # not more than 5× the Laplacian energy
    println("   [graddisc smooth] Laplacian energy ratio = $(round(energy_ratio, digits=2))")
end

# ═══════════════════════════════════════════════════════════════════════════
# 8. Letter E  — glyph-like mixed strokes (text deblurring proxy)
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Letter E: non-blind with horizontal kernel" begin
    sharp = svg_letter_E(128; stroke=4)
    k = motion_kernel(11; θ=0.0)
    _nonblind_test("letterE_horiz", sharp, k; ssim_thresh=0.35)
end

@testset "SVG Letter E: non-blind with curved kernel" begin
    sharp = svg_letter_E(128; stroke=4)
    k = curved_motion_kernel(11; θ=0.0, curvature=0.2)
    _nonblind_test("letterE_curved", sharp, k; ssim_thresh=0.3)
end

@testset "SVG Letter E: blind with horizontal kernel" begin
    sharp = svg_letter_E(128; stroke=4)
    k = motion_kernel(9; θ=0.0)
    _blind_test("letterE_horiz", sharp, k; ssim_thresh=0.01, mse_thresh=0.15)
end

# ═══════════════════════════════════════════════════════════════════════════
# 9. Cross-pattern quality comparison  — same image, three kernel types
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Crosshair: compare linear vs curved vs random-walk (non-blind)" begin
    sharp = svg_crosshair(128; stroke=3)
    k_lin  = motion_kernel(11; θ=0.0)
    k_curv = curved_motion_kernel(11; θ=0.0, curvature=0.3)
    k_rw   = random_walk_kernel(11; seed=42)

    r_lin  = _nonblind_test("compare_lin",  sharp, k_lin;  ssim_thresh=0.35)
    r_curv = _nonblind_test("compare_curv", sharp, k_curv; ssim_thresh=0.15)
    r_rw   = _nonblind_test("compare_rw",   sharp, k_rw;   ssim_thresh=0.25)

    # All three should yield finite, reasonable results
    @test r_lin.ssim  > 0.3
    @test r_curv.ssim > 0.1
    @test r_rw.ssim   > 0.2
    println("   [kernel comparison] linear SSIM=$(round(r_lin.ssim,digits=3))  curved=$(round(r_curv.ssim,digits=3))  random-walk=$(round(r_rw.ssim,digits=3))")
end

# ═══════════════════════════════════════════════════════════════════════════
# 10. Robustness to noise  — checkerboard with additive Gaussian noise
# ═══════════════════════════════════════════════════════════════════════════

@testset "SVG Checkerboard: non-blind with noise (σ=0.01)" begin
    sharp = svg_checkerboard(128; cells=8)
    k = motion_kernel(9; θ=0.0)
    B, _ = make_synthetic_blur(sharp; kernel=k, noise_std=0.01)
    B = Float64.(B)
    dec = deconvolve(B, k; λ=0.008, max_outer=10, max_inner=3)
    dec = Float64.(replace(dec, NaN => 0.0))

    err      = mse(dec, sharp)
    ssim_val = ssim(dec, sharp)

    @test err < 0.08
    @test ssim_val > 0.25
    RobustMotionDeblur.save_composite_summary(
        sharp, B, dec, k, k,
        joinpath(_SVG_FIG_DIR, "checker_noisy_nonblind.png"))
    println("   [checker noisy] MSE=$(round(err,digits=5)) SSIM=$(round(ssim_val,digits=3))")
end

@testset "SVG Checkerboard: non-blind with stronger noise (σ=0.03)" begin
    sharp = svg_checkerboard(128; cells=8)
    k = motion_kernel(9; θ=0.0)
    B, _ = make_synthetic_blur(sharp; kernel=k, noise_std=0.03)
    B = Float64.(B)
    # Higher λ to cope with noise
    dec = deconvolve(B, k; λ=0.02, max_outer=8, max_inner=3)
    dec = Float64.(replace(dec, NaN => 0.0))

    err      = mse(dec, sharp)
    ssim_val = ssim(dec, sharp)

    @test err < 0.10
    @test ssim_val > 0.15
    RobustMotionDeblur.save_composite_summary(
        sharp, B, dec, k, k,
        joinpath(_SVG_FIG_DIR, "checker_noisy_strong_nonblind.png"))
    println("   [checker noisy σ=0.03] MSE=$(round(err,digits=5)) SSIM=$(round(ssim_val,digits=3))")
end
