struct TransformerBlock
    attn_norm::RMSNorm{Float32}
    ffn_norm::RMSNorm{Float32}
    attn
    ffn
end

function TransformerBlock(cfg::MythosConfig; use_moe::Bool=false, rng::AbstractRNG=Random.default_rng())
    attn = cfg.attn_type == "mla" ? MLAttention(cfg; rng=rng) : GQAttention(cfg; rng=rng)
    ffn = use_moe ? MoEFFN(cfg; rng=rng) : Expert(cfg.dim, cfg.dim * 4 ÷ 3; rng=rng)
    return TransformerBlock(RMSNorm(cfg.dim), RMSNorm(cfg.dim), attn, ffn)
end

function (block::TransformerBlock)(x::AbstractArray, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, cache_key::String="default")
    x = x .+ block.attn(block.attn_norm(x), freqs_cis; mask=mask, kv_cache=kv_cache, cache_key=cache_key)
    x = x .+ block.ffn(block.ffn_norm(x))
    return x
end
