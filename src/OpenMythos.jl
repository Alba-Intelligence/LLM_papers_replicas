module OpenMythos

using Base64
using LinearAlgebra
using Random
using Serialization
using Statistics
import Lux
import NNlib
import Optimisers

_sigmoid(x) = inv(one(x) + exp(-x))
_silu(x) = x * _sigmoid(x)

_to_feature_first_perm(n::Int) = (n, (1:(n - 1))...)
_from_feature_first_perm(n::Int) = ((2:n)..., 1)

function _flatten_feature_last(x::AbstractArray)
    ndims(x) >= 2 || throw(ArgumentError("expected at least 2 dimensions"))
    n = ndims(x)
    d = size(x, n)
    return reshape(permutedims(x, _to_feature_first_perm(n)), d, :)
end

function _unflatten_feature_last(flat::AbstractMatrix, refsize::NTuple{N, Int}) where {N}
    n = length(refsize)
    d = refsize[end]
    xperm = reshape(flat, d, refsize[1:(end - 1)]...)
    return permutedims(xperm, _from_feature_first_perm(n))
end

_feature_last_shape(sz::NTuple{N, Int}, outdim::Int) where {N} = (sz[1:(end - 1)]..., outdim)

function _feature_broadcast(v::AbstractVector, nd::Int)
    return reshape(v, ntuple(_ -> 1, nd - 1)..., length(v))
end

function _linear_feature_last(x::AbstractArray, weight::AbstractMatrix, bias::Union{Nothing, AbstractVector}=nothing)
    flat = _flatten_feature_last(x)
    y = weight * flat
    if bias !== nothing
        y .+= reshape(bias, :, 1)
    end
    return _unflatten_feature_last(y, _feature_last_shape(size(x), size(weight, 1)))
end

function _softmax_rows(x::AbstractMatrix)
    out = similar(x)
    for i in axes(x, 1)
        row = @view x[i, :]
        m = maximum(row)
        exps = exp.(row .- m)
        s = sum(exps)
        @view(out[i, :]) .= exps ./ s
    end
    return out
end

function _softmax_cols(x::AbstractMatrix)
    out = similar(x)
    for j in axes(x, 2)
        col = @view x[:, j]
        m = maximum(col)
        exps = exp.(col .- m)
        s = sum(exps)
        @view(out[:, j]) .= exps ./ s
    end
    return out
end

function _embed_tokens(ids::AbstractMatrix{<:Integer}, weight::AbstractMatrix{T}) where {T}
    b, t = size(ids)
    d = size(weight, 2)
    out = Array{T}(undef, b, t, d)
    for i in 1:b, j in 1:t
        idx = ids[i, j] + 1
        (1 <= idx <= size(weight, 1)) || throw(BoundsError(weight, (idx, :)))
        @views out[i, j, :] .= weight[idx, :]
    end
    return out
end

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

include("config.jl")
include("norms.jl")
include("rope.jl")
include("attention.jl")
include("ffn.jl")
include("blocks.jl")
include("recurrent.jl")
include("model.jl")
include("variants.jl")
include("tokenizer.jl")
include("training.jl")

export MythosConfig,
       RMSNorm,
       precompute_rope_freqs,
       apply_rope,
       loop_index_embedding,
       GQAttention,
       MLAttention,
       Expert,
       MoEFFN,
       TransformerBlock,
       RecurrentBlock,
       LoRAAdapter,
       LTIInjection,
       get_A,
       ACTHalting,
       OpenMythos,
       generate,
       DEFAULT_MODEL_ID,
       MythosTokenizer,
       WarmupCosineSchedule,
       learning_rate,
       bootstrap_training_config,
       chunk_next_token_pairs,
       text_next_token_pairs,
       batch_next_token_pairs,
       fineweb_edu_batches,
       LuxHeadOnlyOpenMythos,
       HeadOnlyTrainerState,
       head_only_logits,
       head_only_loss,
       train_head_only_step!,
       train_head_only!,
       latest_checkpoint,
       save_head_only_checkpoint,
       load_head_only_checkpoint,
       mythos_1b,
       mythos_3b,
       mythos_10b,
       mythos_50b,
       mythos_100b,
       mythos_500b,
       mythos_1t

end
