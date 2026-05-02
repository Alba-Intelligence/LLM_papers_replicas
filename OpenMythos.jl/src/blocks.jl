"""
    TransformerBlock

Single OpenMythos transformer block containing attention, normalization, and
either a dense expert or MoE feed-forward stage.
"""
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

"""
    block(x, freqs_cis; mask=nothing, kv_cache=nothing, kv_capacity=nothing, cache_key="default")

Run the transformer block on a feature-last hidden-state tensor.
"""
function (block::TransformerBlock)(x::AbstractArray, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, kv_capacity::Union{Nothing, Integer}=nothing, cache_key::String="default")
    x = x .+ block.attn(block.attn_norm(x), freqs_cis; mask=mask, kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key=cache_key)
    x = x .+ block.ffn(block.ffn_norm(x))
    return x
end
