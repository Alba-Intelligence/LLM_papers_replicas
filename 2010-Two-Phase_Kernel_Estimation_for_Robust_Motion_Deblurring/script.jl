using Pkg

# Use repo project if it has the deps (faster reruns); else temp env for first run
Pkg.activate(temp=true)
repo_project = joinpath(@__DIR__, "Project.toml")
Pkg.develop(path=joinpath(@__DIR__, "packages", "TVL1Deconv"))
Pkg.develop(path=joinpath(@__DIR__, "packages", "RobustMotionDeblur"))
Pkg.add(["Images", "FileIO", "Revise", "ImageFiltering"])


using Revise
using FileIO, Images, RobustMotionDeblur, Statistics

example = "c"

function getFileNames(name)
    l = []
    for t in ["", "_deblur", "_kernel"], ext in ["jpg", "png", "bmp"]
        fn = "scripts/$(name)$(t).$(ext)"
        if isfile(fn)
            push!(l, fn)
        end
    end
    return l
end

# Blur image: use first file that is not a reference output (_deblur, _kernel)
function getBlurPath(name)
    for t in [""], ext in ["jpg", "png", "bmp"]
        fn = "scripts/$(name).$(ext)"
        isfile(fn) && return fn
    end
    return nothing
end

blur_path = getBlurPath(example)
if blur_path === nothing
    # Fallback: synthetic blur so the script always runs and delivers kernel + deblurred image
    println("No scripts/$example.* found. Using synthetic blur (horizontal motion, len=15).")
    using ImageFiltering
    sharp = Gray.(fill(0.1, 200, 200))
    sharp[:, 50:54] .= 0.9
    sharp[:, 98:102] .= 0.9
    sharp[:, 146:150] .= 0.9
    k_synth = RobustMotionDeblur.motion_kernel(15; θ=0.0)
    blur_img = clamp01.(imfilter(sharp, k_synth))
else
    blur_img = load(blur_path)
    println("Blur image: $blur_path")
end
println("Input size: $(size(blur_img))")

println("Running full deblurring pipeline (kernel estimation + TV-L1 deconvolution)...")
deblur_result, estimated_kernel = RobustMotionDeblur.deblur(blur_img)

# Normalize kernel for display: scale by max so structure is visible (not almost black)
k_display = estimated_kernel ./ max(maximum(estimated_kernel), 1e-10)
k_display = clamp01.(replace(k_display, NaN => 0))

println("Done!")
println("Deblurred size: ", size(deblur_result))

# Ensure Float64 and replace NaN
out = Float64.(replace(deblur_result, NaN => 0.0))
lo, hi = minimum(out), maximum(out)
println("Deblurred value range: [$lo, $hi]")

# If we have color output but it's degenerate (nearly same in all channels → single-colour + banding),
# re-deblur on luminance only and colorize from the blurred input so we get sharp structure with original colours.
if ndims(out) == 3 && size(out, 1) == 3
    blur_cv = channelview(blur_img)
    if size(blur_cv, 1) >= 3
        # Check if deblur result is degenerate: channels almost identical (std across channels very low per pixel)
        ch_std = std(out; dims=1)
        mean_ch_std = mean(ch_std)
        if mean_ch_std < 0.02
            println("Deblur result is nearly single-channel (mean cross-channel std = $(round(mean_ch_std, digits=4))). Re-running deconvolution on luminance only and colorizing from input.")
            B_gray = Gray.(blur_img)
            deblur_gray = RobustMotionDeblur.deconvolve(B_gray, estimated_kernel)
            L = Float64.(replace(deblur_gray, NaN => 0.0))
            L = clamp01.(L)
            # Colorize: use deblurred luminance, keep chroma from blurred image (normalize by its luminance to avoid division by zero)
            R_b = Float64.(blur_cv[1, :, :])
            G_b = Float64.(blur_cv[2, :, :])
            B_b = Float64.(blur_cv[3, :, :])
            lum_b = clamp.(0.299 .* R_b .+ 0.587 .* G_b .+ 0.114 .* B_b, 1e-6, 1.0)
            out = similar(out)
            out[1, :, :] = L .* (R_b ./ lum_b)
            out[2, :, :] = L .* (G_b ./ lum_b)
            out[3, :, :] = L .* (B_b ./ lum_b)
            out = clamp01.(out)
        end
    end
end

# Contrast stretch so the image is visible
if ndims(out) == 2
    q = quantile(vec(out), [0.01, 0.99])
    qlo, qhi = q[1], q[2]
    if qhi > qlo + 1e-10
        out = (out .- qlo) ./ (qhi - qlo)
    end
else
    # Color: stretch each channel independently so no single channel dominates (avoids yellow/green cast)
    for c in 1:size(out, 1)
        q = quantile(vec(out[c, :, :]), [0.01, 0.99])
        qlo, qhi = q[1], q[2]
        if qhi > qlo + 1e-10
            out[c, :, :] = (out[c, :, :] .- qlo) ./ (qhi - qlo)
        end
        out[c, :, :] = clamp01.(out[c, :, :])
    end
end
out = clamp01.(out);

println("Kernel size: ", size(estimated_kernel), "  sum: $(round(sum(estimated_kernel), digits=4))")

# Save deblurred image (gray or RGB)
if ndims(out) == 2
    save("result.png", Gray.(out))
else
    save("result.png", colorview(RGB, out))
end

# Save estimated kernel
save("result_kernel.png", Gray.(k_display))
# Single composite: original | blurred | deblurred | actual kernel | estimated kernel (missing = placeholder)
RobustMotionDeblur.save_composite_summary(nothing, blur_img, out, nothing, estimated_kernel, "result_summary.png")
println("Saved result.png, result_kernel.png, result_summary.png")
