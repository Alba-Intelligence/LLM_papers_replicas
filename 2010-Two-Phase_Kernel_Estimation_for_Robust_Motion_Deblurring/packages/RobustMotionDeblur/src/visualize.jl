# Visual summary: single image with [original | blurred | deblurred | actual kernel | estimated kernel].
# Accepts arrays or Images (Gray/RGB); kernels are 2D arrays. Missing panels shown as dark placeholder.

_resize_to(A::AbstractMatrix, (ph, pw)) = [A[clamp(ceil(Int, r * size(A, 1) / ph), 1, size(A, 1)), clamp(ceil(Int, c * size(A, 2) / pw), 1, size(A, 2))] for r in 1:ph, c in 1:pw]

_to_array(x::AbstractArray) = Float64.(x)
_to_array(img) = Float64.(channelview(img))  # Gray, RGB, etc.

_to_gray(x::AbstractMatrix) = clamp01.(x)
_to_gray(x::AbstractArray{<:Number, 3}) = clamp01.(0.299 .* x[1, :, :] .+ 0.587 .* x[2, :, :] .+ 0.114 .* x[3, :, :])

_kernel_display(k) = (k = Float64.(k); m = maximum(k); m > 1e-10 ? k ./ m : k)

"""
    save_composite_summary(original, blurred, deblurred, kernel_true, kernel_est, path; panel_size=(96,96), pad=4)

Write a single image with five panels: original | blurred | deblurred | actual kernel | estimated kernel.
Any argument may be `nothing` (shown as dark placeholder). Images can be 2D, 3D (C×H×W), or Image types (Gray/RGB).
Kernels are 2D arrays (normalized by max for display).
"""
function save_composite_summary(original, blurred, deblurred, kernel_true, kernel_est, path; panel_size=(96, 96), pad=4)
    ph, pw = panel_size
    pad_strip = fill(0.2, ph, pad)
    tile(x) = x === nothing ? fill(0.25, ph, pw) : _resize_to(_to_gray(clamp01.(_to_array(x))), (ph, pw))
    ktile(k) = k === nothing ? fill(0.25, ph, pw) : _resize_to(_kernel_display(k), (ph, pw))
    t1 = tile(original)
    t2 = tile(blurred)
    t3 = tile(deblurred)
    t4 = ktile(kernel_true)
    t5 = ktile(kernel_est)
    row = hcat(t1, pad_strip, t2, pad_strip, t3, pad_strip, t4, pad_strip, t5)
    d = dirname(path)
    if !isempty(d)
        mkpath(d)
    end
    save(path, Gray.(clamp01.(row)))
end
