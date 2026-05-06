module Utils

using Images

export to_gray, grad_x, grad_y, laplacian, normaliseKernel, shrinkKernel

function to_gray(img)
    if isempty(img)
        throw(ArgumentError("Input image is empty"))
    end

    img_size = size(img)
    if length(img_size) < 2
        throw(ArgumentError("Input image must have at least 2 dimensions"))
    end

    if img_size[1] == 0 || img_size[2] == 0
        throw(ArgumentError("Input image has zero size in at least one dimension"))
    end

    A = float.(channelview(img))
    ndims(A) == 2 && return Array(A)

    C = size(A, 1)
    if C >= 3
        Y = 0.299 .* A[1, :, :] .+ 0.587 .* A[2, :, :] .+ 0.114 .* A[3, :, :]
    else
        Y = mean(A; dims = 1)
        Y = dropdims(Y; dims = 1)
    end
    return Array(Y)
end

function normaliseKernel(k)
    k_abs = abs.(k)
    k_sum = sum(k_abs)
    return k_abs ./ (k_sum + eps(Float64))
end

"""
  shrinkKernel(kernel)

Take a kernel and removes all zero-ed rows/columns at the extremes.
The resulting kernel is ensured to have non-zero entries on all sides.
This ensures a better quality of a centered kernel.
"""
function shrinkKernel(kernel)
    sum_by_rows = reshape(sum(abs, kernel, dims = 2), :)
    sum_by_cols = reshape(sum(abs, kernel, dims = 1), :)

    t, b = findfirst(!=(0), sum_by_rows), findlast(!=(0), sum_by_rows)
    l, r = findfirst(!=(0), sum_by_cols), findlast(!=(0), sum_by_cols)

    k_h = b - t + 1
    if iseven(k_h)
        k_h+=1
    end
    k_w = r - l + 1
    if iseven(k_w)
        k_w+=1
    end

    s=zeros(Float64, k_h, k_w)
    s[1:(b-t+1), 1:(r-l+1)] = kernel[t:b, l:r]

    return s
end

function grad_x(I::AbstractMatrix)::Matrix{Float64}
    m, n = size(I)
    gx = zeros(Float64, m, n)
    @inbounds for i in 1:m, j in 1:n
        j2 = clamp(j+1, 1, n)
        gx[i, j] = I[i, j2] - I[i, j]
    end
    return gx
end

function grad_y(I::AbstractMatrix)::Matrix{Float64}
    m, n = size(I)
    gy = zeros(Float64, m, n)
    @inbounds for i in 1:m, j in 1:n
        i2=clamp(i+1, 1, m)
        gy[i, j] = I[i2, j] - I[i, j]
    end
    return gy
end

function laplacian(I::AbstractMatrix)::Matrix{Float64}
    m, n = size(I)
    L = zeros(Float64, m, n)
    @inbounds for i in 1:m, j in 1:n
        c = I[i, j]

        u = I[clamp(i-1, 1, m), j]
        d = I[clamp(i+1, 1, m), j]
        l = I[i, clamp(j-1, 1, n)]
        r = I[i, clamp(j+1, 1, n)]

        L[i, j] = u + d + l + r - 4c
    end
    return L
end

end # module Utils
