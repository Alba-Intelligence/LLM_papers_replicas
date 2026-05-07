"""
    FeatureLinear{T}

Lux-native dense projection for feature-last tensors.
"""
struct FeatureLinear{T<:AbstractFloat} <: Lux.LuxCore.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    use_bias::Bool
    init_scale::T
end

function FeatureLinear(
    in_dim::Integer,
    out_dim::Integer;
    bias::Bool=true,
    init_scale::Real=0.02f0,
    T::Type{<:AbstractFloat}=Float32,
)
    in_dim > 0 || throw(ArgumentError("in_dim must be positive"))
    out_dim > 0 || throw(ArgumentError("out_dim must be positive"))
    return FeatureLinear{T}(Int(in_dim), Int(out_dim), bias, T(init_scale))
end

function Lux.initialparameters(rng::AbstractRNG, layer::FeatureLinear{T}) where {T}
    weight = T.(layer.init_scale .* randn(rng, layer.out_dim, layer.in_dim))
    if layer.use_bias
        return (weight=weight, bias=zeros(T, layer.out_dim))
    end
    return (weight=weight,)
end

Lux.initialstates(::AbstractRNG, ::FeatureLinear) = NamedTuple()

function (layer::FeatureLinear)(x::AbstractArray, ps, st)
    size(x, ndims(x)) == layer.in_dim || throw(DimensionMismatch("expected trailing feature dimension $(layer.in_dim), got $(size(x, ndims(x)))"))
    bias = layer.use_bias ? ps.bias : nothing
    return _linear_feature_last(x, ps.weight, bias), st
end

"""
    TokenEmbedding{T}

Lux-native token embedding lookup layer.
"""
struct TokenEmbedding{T<:AbstractFloat} <: Lux.LuxCore.AbstractLuxLayer
    vocab_size::Int
    dim::Int
    init_scale::T
end

function TokenEmbedding(
    vocab_size::Integer,
    dim::Integer;
    init_scale::Real=0.02f0,
    T::Type{<:AbstractFloat}=Float32,
)
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    dim > 0 || throw(ArgumentError("dim must be positive"))
    return TokenEmbedding{T}(Int(vocab_size), Int(dim), T(init_scale))
end

function Lux.initialparameters(rng::AbstractRNG, layer::TokenEmbedding{T}) where {T}
    return (weight=T.(layer.init_scale .* randn(rng, layer.vocab_size, layer.dim)),)
end

Lux.initialstates(::AbstractRNG, ::TokenEmbedding) = NamedTuple()

function (layer::TokenEmbedding)(ids::AbstractMatrix{<:Integer}, ps, st)
    size(ps.weight) == (layer.vocab_size, layer.dim) || throw(DimensionMismatch("embedding parameter shape does not match layer dimensions"))
    return _embed_tokens(ids, ps.weight), st
end

"""
    tied_lm_head(hidden, embedding_weight)

Project `hidden` into vocabulary logits with a weight matrix that is shared with
an embedding table of shape `(vocab_size, dim)`.
"""
function tied_lm_head(hidden::AbstractArray, embedding_weight::AbstractMatrix)
    size(hidden, ndims(hidden)) == size(embedding_weight, 2) || throw(DimensionMismatch("hidden feature dimension must match embedding width"))
    return _linear_feature_last(hidden, embedding_weight)
end
