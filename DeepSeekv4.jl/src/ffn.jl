"""
    Expert{T}

SwiGLU-style dense expert used by DeepSeek V4 feed-forward blocks.
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

Standard routed Mixture-of-Experts feed-forward block for DeepSeek V4 layers.
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

function MoEFFN(cfg::DeepSeekV4Config; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
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

"""Apply the routed MoE block to a `(batch, time, dim)` tensor."""
function (moe::MoEFFN)(x::AbstractArray{T, 3}) where {T}
    b, t, d = size(x)
    token_matrix = reshape(permutedims(x, (3, 1, 2)), d, :)'
    n = size(token_matrix, 1)

    logits = moe.router * transpose(token_matrix)
    scores = _softmax_cols(logits)
    token_outputs = [
        begin
            adjusted = view(logits, :, token_idx) .+ moe.router_bias
            top_idx = partialsortperm(vec(adjusted), 1:moe.topk; rev=true)
            token_scores = scores[top_idx, token_idx]
            token_scores = token_scores ./ sum(token_scores)
            token = vec(@view token_matrix[token_idx, :])
            reduce(
                .+,
                (score .* moe.routed_experts[expert_idx](token) for (score, expert_idx) in zip(token_scores, top_idx));
                init=zeros(T, d),
            )
        end
        for token_idx in 1:n
    ]
    out = cat([reshape(token_out, 1, d) for token_out in token_outputs]...; dims=1)

    for shared in moe.shared_experts
        out = out .+ shared(token_matrix)
    end

    return permutedims(reshape(out, b, t, d), (1, 2, 3))
end

"""
    HashMoEFFN{T}

Hash-routed MoE variant used in the early DeepSeek V4 layers.
"""
struct HashMoEFFN{T<:AbstractFloat}
    n_experts::Int
    n_shared::Int
    topk::Int
    layer_seed::Int
    routed_experts::Vector{Expert{T}}
    shared_experts::Vector{Expert{T}}
end

function HashMoEFFN(
    n_experts::Int,
    n_shared::Int,
    topk::Int,
    layer_seed::Int,
    routed_experts::AbstractVector{<:Expert{T}},
    shared_experts::AbstractVector,
) where {T<:AbstractFloat}
    return HashMoEFFN{T}(
        n_experts,
        n_shared,
        topk,
        layer_seed,
        Vector{Expert{T}}(routed_experts),
        Vector{Expert{T}}(shared_experts),
    )
end

function HashMoEFFN(cfg::DeepSeekV4Config, layer_seed::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    routed = [Expert(cfg.dim, cfg.expert_dim; rng=rng, T=T) for _ in 1:cfg.n_experts]
    shared = cfg.n_shared_experts == 0 ?
        Expert{T}[] :
        [Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok; rng=rng, T=T) for _ in 1:cfg.n_shared_experts]
    return HashMoEFFN(cfg.n_experts, cfg.n_shared_experts, cfg.n_experts_per_tok, Int(layer_seed), routed, shared)
end

function _next_unique_hash_candidate(candidate::Int, chosen::Tuple, n_experts::Int)
    candidate in chosen || return candidate
    return _next_unique_hash_candidate(mod(candidate, n_experts) + 1, chosen, n_experts)
end

function _hash_expert_indices_tuple(base::Int, n_experts::Int, topk::Int, layer_seed::Int, offset::Int, chosen::Tuple)
    length(chosen) == topk && return chosen
    candidate = mod(base * 1_103_515_245 + (layer_seed + 1) * 12_345 + offset * 97, n_experts) + 1
    candidate = _next_unique_hash_candidate(candidate, chosen, n_experts)
    return _hash_expert_indices_tuple(base, n_experts, topk, layer_seed, offset + 1, (chosen..., candidate))
end

function _hash_expert_indices(token_id::Integer, n_experts::Int, topk::Int, layer_seed::Int)
    n_experts > 0 || throw(ArgumentError("n_experts must be positive"))
    topk > 0 || throw(ArgumentError("topk must be positive"))
    topk <= n_experts || throw(ArgumentError("topk must be <= n_experts"))
    return collect(_hash_expert_indices_tuple(Int(token_id), n_experts, topk, layer_seed, 0, ()))
end

"""Apply the hash-routed MoE block using `token_ids` to choose routed experts."""
function (moe::HashMoEFFN)(x::AbstractArray{T, 3}, token_ids::AbstractMatrix{<:Integer}) where {T}
    size(x, 1) == size(token_ids, 1) || throw(DimensionMismatch("batch size mismatch"))
    size(x, 2) == size(token_ids, 2) || throw(DimensionMismatch("sequence length mismatch"))
    b, t, d = size(x)
    token_outputs = [
        begin
            token = vec(@view x[bi, ti, :])
            selected = _hash_expert_indices(token_ids[bi, ti], moe.n_experts, moe.topk, moe.layer_seed)
            weight = inv(T(length(selected)))
            reshape(
                reduce(
                    .+,
                    (weight .* moe.routed_experts[expert_idx](token) for expert_idx in selected);
                    init=zeros(T, d),
                ),
                1,
                1,
                d,
            )
        end
        for bi in 1:b, ti in 1:t
    ]
    out = cat([cat([token_outputs[bi, ti] for ti in 1:t]...; dims=2) for bi in 1:b]...; dims=1)
    for shared in moe.shared_experts
        out = out .+ shared(x)
    end
    return out
end
