function precompute_rope_freqs(dim::Integer, max_len::Integer; theta::Real=500_000.0f0)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    iseven(dim) || throw(ArgumentError("dim must be even"))
    max_len > 0 || throw(ArgumentError("max_len must be positive"))

    work_t = Float32(theta)
    freq_idx = Float32.(0:2:(dim - 2))
    base = 1.0f0 ./ (work_t .^ (freq_idx ./ Float32(dim)))
    positions = Float32.(0:(max_len - 1))
    angles = positions .* transpose(base)
    return ComplexF32.(cos.(angles), sin.(angles))
end

function apply_rope(x::AbstractArray{T, 4}, freqs_cis::AbstractMatrix{<:Complex}) where {T<:AbstractFloat}
    b, t, h, d = size(x)
    iseven(d) || throw(ArgumentError("head dimension must be even"))
    t <= size(freqs_cis, 1) || throw(DimensionMismatch("sequence length exceeds precomputed RoPE table"))
    size(freqs_cis, 2) == d ÷ 2 || throw(DimensionMismatch("RoPE table width must equal head_dim ÷ 2"))

    work = Float32.(x)
    d2 = d ÷ 2
    cosvals = reshape(Float32.(real(freqs_cis[1:t, :])), 1, t, 1, d2)
    sinvals = reshape(Float32.(imag(freqs_cis[1:t, :])), 1, t, 1, d2)

    x1 = @view work[:, :, :, 1:2:d]
    x2 = @view work[:, :, :, 2:2:d]
    out = similar(work)
    @views out[:, :, :, 1:2:d] .= x1 .* cosvals .- x2 .* sinvals
    @views out[:, :, :, 2:2:d] .= x1 .* sinvals .+ x2 .* cosvals

    return T.(out)
end
