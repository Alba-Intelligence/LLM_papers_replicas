"""
    OLMoMLP{T}

Bias-free SwiGLU feed-forward block used by OLMo-family decoder layers.
"""
struct OLMoMLP{T<:AbstractFloat}
    gate_proj::Matrix{T}
    up_proj::Matrix{T}
    down_proj::Matrix{T}
end

function OLMoMLP(cfg::OLMoConfig; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    _validate_olmo_cfg(cfg)
    return OLMoMLP(
        T.(cfg.init_scale .* randn(rng, cfg.intermediate_dim, cfg.dim)),
        T.(cfg.init_scale .* randn(rng, cfg.intermediate_dim, cfg.dim)),
        T.(cfg.init_scale .* randn(rng, cfg.dim, cfg.intermediate_dim)),
    )
end

function (mlp::OLMoMLP)(x::AbstractArray{T, 3}) where {T<:AbstractFloat}
    gate = _linear_feature_last(x, mlp.gate_proj)
    up = _linear_feature_last(x, mlp.up_proj)
    hidden = _silu.(gate) .* up
    return _linear_feature_last(hidden, mlp.down_proj)
end

"""
    OLMoBlock

Single OLMo-family decoder block with QK-Norm attention, SwiGLU MLP, and the
inside-residual post-norm layout used by OLMo 2.
"""
struct OLMoBlock{T<:AbstractFloat}
    layer_index::Int
    attn::OLMoAttention{T}
    mlp::OLMoMLP{T}
    post_attention_norm::RMSNorm{T}
    post_feedforward_norm::RMSNorm{T}
end

function OLMoBlock(cfg::OLMoConfig, layer_index::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    _validate_olmo_cfg(cfg)
    return OLMoBlock(
        Int(layer_index),
        OLMoAttention(cfg; rng=rng, T=T),
        OLMoMLP(cfg; rng=rng, T=T),
        RMSNorm(cfg.dim; eps=cfg.rms_norm_eps, T=T),
        RMSNorm(cfg.dim; eps=cfg.rms_norm_eps, T=T),
    )
end

"""
    block(x, freqs_cis; mask=nothing, kv_cache=nothing, kv_capacity=nothing, cache_key="default")

Apply one OLMo-family decoder block to feature-last hidden states.
"""
function (block::OLMoBlock{T})(x::AbstractArray{T, 3}, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, kv_capacity::Union{Nothing, Integer}=nothing, cache_key::String="default") where {T<:AbstractFloat}
    residual = x
    attn_out = block.attn(x, freqs_cis; mask=mask, kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key=cache_key)
    x = residual .+ block.post_attention_norm(attn_out)

    residual = x
    mlp_out = block.mlp(x)
    return residual .+ block.post_feedforward_norm(mlp_out)
end
