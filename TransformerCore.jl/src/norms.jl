"""
    RMSNorm{T}

Root-mean-square normalization with a learned feature-wise scale.
"""
struct RMSNorm{T<:AbstractFloat} <: Lux.LuxCore.AbstractLuxLayer
    weight::Vector{T}
    eps::T
end

function RMSNorm(dim::Integer; eps::Real=1f-6, T::Type{<:AbstractFloat}=Float32)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    return RMSNorm(fill(one(T), dim), T(eps))
end

"""
    norm(x)

Apply RMS normalization across the last dimension of `x`.
"""
function (norm::RMSNorm)(x::AbstractArray)
    length(norm.weight) == size(x, ndims(x)) || throw(DimensionMismatch("weight size must match last dimension"))
    scale = inv.(sqrt.(mean(abs2, x; dims=ndims(x)) .+ norm.eps))
    return x .* scale .* _feature_broadcast(norm.weight, ndims(x))
end

function Lux.initialparameters(::AbstractRNG, norm::RMSNorm)
    return (weight=copy(norm.weight),)
end

Lux.initialstates(::AbstractRNG, ::RMSNorm) = NamedTuple()

function (norm::RMSNorm)(x::AbstractArray, ps, st)
    length(ps.weight) == size(x, ndims(x)) || throw(DimensionMismatch("weight size must match last dimension"))
    scale = inv.(sqrt.(mean(abs2, x; dims=ndims(x)) .+ norm.eps))
    return x .* scale .* _feature_broadcast(ps.weight, ndims(x)), st
end
