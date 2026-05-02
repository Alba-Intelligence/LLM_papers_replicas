"""
    Expert{T}

SwiGLU-style feed-forward expert used by OpenMythos.
"""
struct Expert{T<:AbstractFloat}
    gate::Matrix{T}
    up::Matrix{T}
    down::Matrix{T}
end

function Expert(dim::Integer, expert_dim::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    return Expert(
        T.(0.02 .* randn(rng, expert_dim, dim)),
        T.(0.02 .* randn(rng, expert_dim, dim)),
        T.(0.02 .* randn(rng, dim, expert_dim)),
    )
end

"""Apply an `Expert` to a single feature vector."""
function (expert::Expert)(x::AbstractVector)
    gate = expert.gate * x
    up = expert.up * x
    return expert.down * (_silu.(gate) .* up)
end

"""Apply an `Expert` to a feature-last tensor."""
function (expert::Expert)(x::AbstractArray)
    gate = _linear_feature_last(x, expert.gate)
    up = _linear_feature_last(x, expert.up)
    return _linear_feature_last(_silu.(gate) .* up, expert.down)
end

"""
    MoEFFN{T}

Mixture-of-experts feed-forward network used inside the recurrent OpenMythos block.
"""
struct MoEFFN{T<:AbstractFloat}
    n_experts::Int
    n_shared::Int
    topk::Int
    router::Matrix{T}
    router_bias::Vector{T}
    routed_experts::Vector{Expert{T}}
    shared_experts::Vector{Expert{T}}
end

function MoEFFN(
    n_experts::Int,
    n_shared::Int,
    topk::Int,
    router::Matrix{T},
    router_bias::Vector{T},
    routed_experts::AbstractVector,
    shared_experts::AbstractVector,
) where {T<:AbstractFloat}
    return MoEFFN{T}(
        n_experts,
        n_shared,
        topk,
        router,
        router_bias,
        Vector{Expert{T}}(routed_experts),
        Vector{Expert{T}}(shared_experts),
    )
end

function MoEFFN(cfg::MythosConfig; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    routed = [Expert(cfg.dim, cfg.expert_dim; rng=rng, T=T) for _ in 1:cfg.n_experts]
    shared = cfg.n_shared_experts == 0 ?
        Expert{T}[] :
        [Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok; rng=rng, T=T) for _ in 1:cfg.n_shared_experts]
    return MoEFFN(
        cfg.n_experts,
        cfg.n_shared_experts,
        cfg.n_experts_per_tok,
        T.(0.02 .* randn(rng, cfg.n_experts, cfg.dim)),
        zeros(T, cfg.n_experts),
        routed,
        shared,
    )
end

"""Apply the MoE feed-forward block to a `(batch, time, dim)` tensor."""
function (moe::MoEFFN)(x::AbstractArray{T, 3}) where {T}
    if moe.n_experts == 1 && moe.topk == 1
        out = moe.routed_experts[1](x)
        for shared in moe.shared_experts
            out = out .+ shared(x)
        end
        return out
    end

    b, t, d = size(x)
    token_matrix = reshape(permutedims(x, (3, 1, 2)), d, :)'
    n = size(token_matrix, 1)

    logits = moe.router * transpose(token_matrix)
    scores = _softmax_cols(logits)
    out = zeros(T, n, d)

    for token_idx in 1:n
        adjusted = view(logits, :, token_idx) .+ moe.router_bias
        top_idx = partialsortperm(vec(adjusted), 1:moe.topk; rev=true)
        token_scores = scores[top_idx, token_idx]
        token_scores ./= sum(token_scores)
        token = vec(@view token_matrix[token_idx, :])
        for (score, expert_idx) in zip(token_scores, top_idx)
            out[token_idx, :] .+= score .* moe.routed_experts[expert_idx](token)
        end
    end

    for shared in moe.shared_experts
        out .+= shared(token_matrix)
    end

    return permutedims(reshape(out, b, t, d), (1, 2, 3))
end
