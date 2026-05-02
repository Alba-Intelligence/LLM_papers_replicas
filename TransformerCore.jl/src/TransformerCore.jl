"""
    TransformerCore

Shared tensor, normalization, scheduling, batching, and cache helpers used by the
`OpenMythos.jl` and `DeepSeekv4.jl` packages.

The module intentionally stays architecture-agnostic: it provides feature-last
linear algebra helpers, RoPE utilities, schedule/batching helpers, and the shared
KV-cache envelope/buffer layer without encoding model-family-specific behavior.
"""
module TransformerCore

using LinearAlgebra
using Random
using Serialization
using Statistics

"""Compute the logistic sigmoid of `x`."""
_sigmoid(x) = inv(one(x) + exp(-x))

"""Compute the SiLU/Swish activation of `x`."""
_silu(x) = x * _sigmoid(x)

_to_feature_first_perm(n::Int) = (n, (1:(n - 1))...)
_from_feature_first_perm(n::Int) = ((2:n)..., 1)

"""
    _flatten_feature_last(x)

Reshape a feature-last tensor into a matrix of shape `(feature_dim, n_items)`.

This is the core layout transform used throughout the workspace to implement
feature-last projections with ordinary matrix multiplication.
"""
function _flatten_feature_last(x::AbstractArray)
    ndims(x) >= 2 || throw(ArgumentError("expected at least 2 dimensions"))
    n = ndims(x)
    d = size(x, n)
    return reshape(permutedims(x, _to_feature_first_perm(n)), d, :)
end

"""
    _unflatten_feature_last(flat, refsize)

Undo `_flatten_feature_last` using the feature-last reference shape `refsize`.
"""
function _unflatten_feature_last(flat::AbstractMatrix, refsize::NTuple{N, Int}) where {N}
    n = length(refsize)
    d = refsize[end]
    xperm = reshape(flat, d, refsize[1:(end - 1)]...)
    return permutedims(xperm, _from_feature_first_perm(n))
end

"""
    _feature_last_shape(sz, outdim)

Return the feature-last output shape obtained by replacing the final dimension of
`sz` with `outdim`.
"""
_feature_last_shape(sz::NTuple{N, Int}, outdim::Int) where {N} = (sz[1:(end - 1)]..., outdim)

"""
    _feature_broadcast(v, nd)

Reshape a feature vector `v` so it can broadcast across an `nd`-dimensional
feature-last tensor.
"""
function _feature_broadcast(v::AbstractVector, nd::Int)
    return reshape(v, ntuple(_ -> 1, nd - 1)..., length(v))
end

"""
    _linear_feature_last(x, weight[, bias])

Apply a dense linear map to a feature-last tensor `x`.

`weight` is expected to have shape `(out_dim, in_dim)`. When `bias` is provided,
it is broadcast across all non-feature dimensions.
"""
function _linear_feature_last(x::AbstractArray, weight::AbstractMatrix, bias::Union{Nothing, AbstractVector}=nothing)
    flat = _flatten_feature_last(x)
    y = bias === nothing ? (weight * flat) : ((weight * flat) .+ reshape(bias, :, 1))
    return _unflatten_feature_last(y, _feature_last_shape(size(x), size(weight, 1)))
end

"""Apply a row-wise softmax to `x`."""
function _softmax_rows(x::AbstractMatrix)
    shifted = x .- maximum(x; dims=2)
    exps = exp.(shifted)
    return exps ./ sum(exps; dims=2)
end

"""Apply a column-wise softmax to `x`."""
function _softmax_cols(x::AbstractMatrix)
    shifted = x .- maximum(x; dims=1)
    exps = exp.(shifted)
    return exps ./ sum(exps; dims=1)
end

"""
    _embed_tokens(ids, weight)

Lookup token embeddings from `weight` and return a tensor with shape
`(batch, time, dim)`.
"""
function _embed_tokens(ids::AbstractMatrix{<:Integer}, weight::AbstractMatrix{T}) where {T}
    b, t = size(ids)
    idx = vec(ids) .+ 1
    all((1 .<= idx) .& (idx .<= size(weight, 1))) || throw(BoundsError(weight, (idx, :)))
    rows = weight[idx, :]
    return permutedims(reshape(permutedims(rows, (2, 1)), size(weight, 2), b, t), (2, 3, 1))
end

"""
    _sample_categorical(probs; rng=Random.default_rng())

Sample one token id per row from a matrix of categorical probabilities.

Returns a `(batch, 1)` matrix of zero-based token ids.
"""
function _sample_categorical(probs::AbstractMatrix{T}; rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    b, v = size(probs)
    out = Array{Int}(undef, b, 1)
    for i in 1:b
        r = rand(rng, T)
        acc = zero(T)
        chosen = v
        for j in 1:v
            acc += probs[i, j]
            if r <= acc
                chosen = j
                break
            end
        end
        out[i, 1] = chosen - 1
    end
    return out
end

include("norms.jl")
include("rope.jl")
include("training_utils.jl")
include("cache_buffers.jl")
include("kv_cache.jl")

export _sigmoid,
       _silu,
       _flatten_feature_last,
       _unflatten_feature_last,
       _feature_last_shape,
       _feature_broadcast,
       _linear_feature_last,
       _softmax_rows,
       _softmax_cols,
       _embed_tokens,
       _sample_categorical,
       RMSNorm,
       precompute_rope_freqs,
       apply_rope,
       WarmupCosineSchedule,
       learning_rate,
       chunk_next_token_pairs,
       text_next_token_pairs,
       batch_next_token_pairs,
       latest_checkpoint,
       _head_loss_and_grad,
       buffer_capacity,
       buffer_page_count,
       ensure_axis_capacity!,
       KVCacheEnvelope,
       reserve_kv_capacity!,
       save_kv_cache,
       load_kv_cache

end
