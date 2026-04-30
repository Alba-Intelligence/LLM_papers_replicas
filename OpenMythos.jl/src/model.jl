struct OpenMythos{T<:AbstractFloat}
    cfg::MythosConfig
    embed::Matrix{T}
    freqs_cis::Matrix{ComplexF32}
    freqs_cis_mla::Matrix{ComplexF32}
    prelude::Vector{TransformerBlock}
    recurrent::RecurrentBlock
    coda::Vector{TransformerBlock}
    norm::RMSNorm{T}
    head::Matrix{T}
end

function OpenMythos(cfg::MythosConfig; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    embed = T.(0.02 .* randn(rng, cfg.vocab_size, cfg.dim))
    prelude = [TransformerBlock(cfg; rng=rng) for _ in 1:cfg.prelude_layers]
    coda = [TransformerBlock(cfg; rng=rng) for _ in 1:cfg.coda_layers]
    return OpenMythos(
        cfg,
        embed,
        precompute_rope_freqs(cfg.dim ÷ cfg.n_heads, cfg.max_seq_len; theta=cfg.rope_theta),
        precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len; theta=cfg.rope_theta),
        prelude,
        RecurrentBlock(cfg; rng=rng),
        coda,
        RMSNorm(cfg.dim; T=T),
        embed,
    )
end

function _forward_hidden(model::OpenMythos, input_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0)
    t = size(input_ids, 2)
    x = _embed_tokens(input_ids, model.embed)
    freqs_all = model.cfg.attn_type == "mla" ? model.freqs_cis_mla : model.freqs_cis
    freqs = freqs_all[(start_pos + 1):(start_pos + t), :]
    mask = t > 1 ? _causal_mask(t, start_pos, Float32) : nothing

    for (i, layer) in enumerate(model.prelude)
        x = layer(x, freqs; mask=mask, kv_cache=kv_cache, cache_key="prelude_$(i - 1)")
    end

    e = x
    x = model.recurrent(x, e, freqs; mask=mask, n_loops=n_loops, kv_cache=kv_cache)

    for (i, layer) in enumerate(model.coda)
        x = layer(x, freqs; mask=mask, kv_cache=kv_cache, cache_key="coda_$(i - 1)")
    end

    return model.norm(x)
end

function (model::OpenMythos)(input_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0)
    hidden = _forward_hidden(model, input_ids; n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
    return _linear_feature_last(hidden, model.head)
end

function chunked_prefill(
    model::OpenMythos,
    input_ids::AbstractMatrix{<:Integer};
    chunk_size::Integer,
    n_loops::Union{Nothing, Integer}=nothing,
    envelope::KVCacheEnvelope=KVCacheEnvelope(),
)
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    size(input_ids, 2) <= 1 && return envelope

    prefix = @view input_ids[:, 1:(end - 1)]
    total_t = size(prefix, 2)
    start = 1
    while start <= total_t
        stop = min(start + Int(chunk_size) - 1, total_t)
        chunk = @view prefix[:, start:stop]
        _forward_hidden(model, chunk; n_loops=n_loops, kv_cache=envelope.cache, start_pos=envelope.start_pos)
        envelope.start_pos += size(chunk, 2)
        start = stop + 1
    end
    return envelope
end

function generate(model::OpenMythos, input_ids::AbstractMatrix{<:Integer}; max_new_tokens::Integer=64, n_loops::Integer=8, temperature::Real=1.0, top_k::Integer=50, rng::AbstractRNG=Random.default_rng(), envelope::Union{Nothing, KVCacheEnvelope}=nothing)
    ids = copy(input_ids)
    kv_cache = envelope === nothing ? Dict{String, Any}() : envelope.cache
    has_prefill = envelope !== nothing && (!isempty(envelope.cache) || envelope.start_pos > 0)
    prompt_len = size(ids, 2)

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

        logits = model(cur_ids; n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
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
