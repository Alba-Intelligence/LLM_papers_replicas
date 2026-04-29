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
    mask = t > 1 ? _causal_mask(t, Float32) : nothing

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

function generate(model::OpenMythos, input_ids::AbstractMatrix{<:Integer}; max_new_tokens::Integer=64, n_loops::Integer=8, temperature::Real=1.0, top_k::Integer=50, rng::AbstractRNG=Random.default_rng())
    ids = copy(input_ids)
    kv_cache = Dict{String, Any}()
    prompt_len = size(ids, 2)

    for step in 1:max_new_tokens
        if step == 1
            cur_ids = ids
            start_pos = 0
        else
            cur_ids = ids[:, end:end]
            start_pos = prompt_len + step - 2
        end

        logits = model(cur_ids; n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
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
