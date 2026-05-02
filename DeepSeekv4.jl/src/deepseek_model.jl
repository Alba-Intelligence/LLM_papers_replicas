struct DeepSeekV4Block
    layer_index::Int
    mix::ManifoldHyperConnections{Float32}
    attn_norm::RMSNorm{Float32}
    ffn_norm::RMSNorm{Float32}
    attn
    ffn
end

function _deepseek_attention_kind(cfg::DeepSeekV4Config, layer_index::Integer)
    return cfg.attention_pattern[mod1(Int(layer_index), length(cfg.attention_pattern))]
end

function DeepSeekV4Block(cfg::DeepSeekV4Config, layer_index::Integer; rng::AbstractRNG=Random.default_rng())
    attn = _deepseek_attention_kind(cfg, layer_index) === :hca ? HeavilyCompressedAttention(cfg; rng=rng) : CompressedSparseAttention(cfg; rng=rng)
    ffn = layer_index <= cfg.hash_routed_layers ? HashMoEFFN(cfg, layer_index; rng=rng) : MoEFFN(cfg; rng=rng)
    return DeepSeekV4Block(
        Int(layer_index),
        ManifoldHyperConnections(cfg.dim, cfg.n_hyper_connections; rng=rng),
        RMSNorm(cfg.dim),
        RMSNorm(cfg.dim),
        attn,
        ffn,
    )
end

function (block::DeepSeekV4Block)(X::AbstractArray{T, 4}, token_ids::AbstractMatrix{<:Integer}, freqs_cis::AbstractMatrix; kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0, kv_capacity::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    cache_key = "deepseek_layer_$(block.layer_index - 1)"
    return block.mix(X, x -> begin
        h = x .+ block.attn(block.attn_norm(x), freqs_cis; kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key=cache_key, start_pos=start_pos)
        if block.ffn isa HashMoEFFN
            return h .+ block.ffn(block.ffn_norm(h), token_ids)
        end
        return h .+ block.ffn(block.ffn_norm(h))
    end)
end

struct DeepSeekV4Model{T<:AbstractFloat}
    cfg::DeepSeekV4Config
    embed::Matrix{T}
    freqs_cis::Matrix{ComplexF32}
    blocks::Vector{DeepSeekV4Block}
    readout::ManifoldHyperConnections{T}
    norm::RMSNorm{T}
    head::Matrix{T}
    mtp_heads::Vector{Matrix{T}}
end

function DeepSeekV4Model(cfg::DeepSeekV4Config; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    embed = T.(0.02 .* randn(rng, cfg.vocab_size, cfg.dim))
    blocks = [DeepSeekV4Block(cfg, layer_idx; rng=rng) for layer_idx in 1:cfg.n_layers]
    mtp_heads = [T.(0.02 .* randn(rng, cfg.vocab_size, cfg.dim)) for _ in 1:cfg.mtp_tokens]
    return DeepSeekV4Model(
        cfg,
        embed,
        precompute_rope_freqs(_rope_suffix_dim(cfg.dim ÷ cfg.n_heads), cfg.max_seq_len; theta=cfg.rope_theta),
        blocks,
        ManifoldHyperConnections(cfg.dim, cfg.n_hyper_connections; rng=rng, T=T, sinkhorn_iters=cfg.sinkhorn_iters),
        RMSNorm(cfg.dim; T=T),
        embed,
        mtp_heads,
    )
end

function _deepseek_init_state(x::AbstractArray{T, 3}, n_streams::Integer) where {T<:AbstractFloat}
    b, t, d = size(x)
    X = zeros(T, b, t, Int(n_streams), d)
    @views X[:, :, 1, :] .= x
    return X
end

function deepseek_hidden(model::DeepSeekV4Model, input_ids::AbstractMatrix{<:Integer}; kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0, kv_capacity::Union{Nothing, Integer}=nothing)
    t = size(input_ids, 2)
    x = _embed_tokens(input_ids, model.embed)
    freqs = model.freqs_cis[(start_pos + 1):(start_pos + t), :]
    X = _deepseek_init_state(x, model.cfg.n_hyper_connections)
    for block in model.blocks
        X = block(X, input_ids, freqs; kv_cache=kv_cache, start_pos=start_pos, kv_capacity=kv_capacity)
    end
    return model.norm(mhc_readout(model.readout, X))
end

function (model::DeepSeekV4Model)(input_ids::AbstractMatrix{<:Integer}; kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0, kv_capacity::Union{Nothing, Integer}=nothing)
    hidden = deepseek_hidden(model, input_ids; kv_cache=kv_cache, start_pos=start_pos, kv_capacity=kv_capacity)
    return _linear_feature_last(hidden, model.head)
end

function mtp_logits(model::DeepSeekV4Model, input_ids::AbstractMatrix{<:Integer}; kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0, kv_capacity::Union{Nothing, Integer}=nothing)
    hidden = deepseek_hidden(model, input_ids; kv_cache=kv_cache, start_pos=start_pos, kv_capacity=kv_capacity)
    outputs = [_linear_feature_last(hidden, head) for head in model.mtp_heads]
    return cat([reshape(out, size(out, 1), size(out, 2), 1, size(out, 3)) for out in outputs]...; dims=3)
end

function chunked_prefill(
    model::DeepSeekV4Model,
    input_ids::AbstractMatrix{<:Integer};
    chunk_size::Integer,
    envelope::KVCacheEnvelope=KVCacheEnvelope(),
)
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    size(input_ids, 2) <= 1 && return envelope

    prefix = @view input_ids[:, 1:(end - 1)]
    total_t = size(prefix, 2)
    reserve_kv_capacity!(envelope, envelope.start_pos + total_t)
    start = 1
    while start <= total_t
        stop = min(start + Int(chunk_size) - 1, total_t)
        chunk = @view prefix[:, start:stop]
        deepseek_hidden(model, chunk; kv_cache=envelope.cache, start_pos=envelope.start_pos, kv_capacity=envelope.capacity_hint)
        envelope.start_pos += size(chunk, 2)
        start = stop + 1
    end
    return envelope
end

function generate(model::DeepSeekV4Model, input_ids::AbstractMatrix{<:Integer}; max_new_tokens::Integer=64, temperature::Real=1.0, top_k::Integer=50, rng::AbstractRNG=Random.default_rng(), envelope::Union{Nothing, KVCacheEnvelope}=nothing)
    ids = copy(input_ids)
    kv_cache = envelope === nothing ? Dict{String, Any}() : envelope.cache
    has_prefill = envelope !== nothing && (!isempty(envelope.cache) || envelope.start_pos > 0)
    prompt_len = size(ids, 2)
    envelope !== nothing && reserve_kv_capacity!(envelope, prompt_len + Int(max_new_tokens) - 1)
    for step in 1:max_new_tokens
        if step == 1
            if has_prefill
                cur_ids = ids[:, end:end]
                start_pos = envelope.start_pos
            else
                cur_ids = ids
                start_pos = 0
            end
        else
            cur_ids = ids[:, end:end]
            start_pos = prompt_len + step - 2
        end
        logits = model(cur_ids; kv_cache=kv_cache, start_pos=start_pos, kv_capacity=(envelope === nothing ? nothing : envelope.capacity_hint))
        envelope !== nothing && (envelope.start_pos = start_pos + size(cur_ids, 2))
        logits = logits[:, end, :] ./ Float32(temperature)
        if top_k > 0 && top_k < size(logits, 2)
            filtered = copy(logits)
            for i in axes(filtered, 1)
                idx = partialsortperm(vec(filtered[i, :]), 1:top_k; rev=true)
                keep = falses(size(filtered, 2))
                keep[idx] .= true
                filtered[i, .!keep] .= -Inf32
            end
            logits = filtered
        end
        probs = _softmax_rows(logits)
        next_tok = _sample_categorical(probs; rng=rng)
        ids = cat(ids, next_tok; dims=2)
    end
    return ids
end
