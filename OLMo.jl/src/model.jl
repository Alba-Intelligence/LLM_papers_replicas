"""
    OLMoModel{T}

Dense decoder-only OLMo-family language model.
"""
struct OLMoModel{T<:AbstractFloat}
    cfg::OLMoConfig
    embed::Matrix{T}
    freqs_cis::Matrix{ComplexF32}
    blocks::Vector{OLMoBlock{T}}
    norm::RMSNorm{T}
    head::Matrix{T}
end

function OLMoModel(cfg::OLMoConfig; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    _validate_olmo_cfg(cfg)
    embed = T.(cfg.init_scale .* randn(rng, cfg.vocab_size, cfg.dim))
    head = T.(cfg.init_scale .* randn(rng, cfg.vocab_size, cfg.dim))
    blocks = [OLMoBlock(cfg, layer_idx; rng=rng, T=T) for layer_idx in 1:cfg.n_layers]
    return OLMoModel(
        cfg,
        embed,
        precompute_rope_freqs(cfg.dim ÷ cfg.n_heads, cfg.max_seq_len; theta=cfg.rope_theta),
        blocks,
        RMSNorm(cfg.dim; eps=cfg.rms_norm_eps, T=T),
        head,
    )
end

"""
    olmo_hidden(model, input_ids; kv_cache=nothing, start_pos=0, kv_capacity=nothing)

Return the final hidden states produced by `model` for `input_ids`.
"""
function olmo_hidden(model::OLMoModel{T}, input_ids::AbstractMatrix{<:Integer}; kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0, kv_capacity::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    start_pos >= 0 || throw(ArgumentError("start_pos must be non-negative"))
    t = size(input_ids, 2)
    x = _embed_tokens(input_ids, model.embed)
    freqs = model.freqs_cis[(start_pos + 1):(start_pos + t), :]
    mask = t > 1 ? _causal_mask(t, start_pos, T) : nothing
    for (i, block) in enumerate(model.blocks)
        x = block(x, freqs; mask=mask, kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key="layer_$(i - 1)")
    end
    return model.norm(x)
end

"""
    model(input_ids; kv_cache=nothing, start_pos=0, kv_capacity=nothing)

Run the OLMo model forward and return token logits.
"""
function (model::OLMoModel{T})(input_ids::AbstractMatrix{<:Integer}; kv_cache::Union{Nothing, AbstractDict}=nothing, start_pos::Integer=0, kv_capacity::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    hidden = olmo_hidden(model, input_ids; kv_cache=kv_cache, start_pos=start_pos, kv_capacity=kv_capacity)
    return _linear_feature_last(hidden, model.head)
end

"""
    chunked_prefill(model, input_ids; chunk_size, envelope=KVCacheEnvelope())

Run a prompt prefix through the model in chunks and accumulate the resulting
cache state into `envelope`.
"""
function chunked_prefill(
    model::OLMoModel,
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
        olmo_hidden(model, chunk; kv_cache=envelope.cache, start_pos=envelope.start_pos, kv_capacity=envelope.capacity_hint)
        envelope.start_pos += size(chunk, 2)
        start = stop + 1
    end
    return envelope
end

"""
    generate(model, input_ids; max_new_tokens=64, temperature=1.0, top_k=50, rng=Random.default_rng(), envelope=nothing)

Autoregressively sample continuations from `model`, optionally resuming from a
prefilled `KVCacheEnvelope`.
"""
function generate(model::OLMoModel, input_ids::AbstractMatrix{<:Integer}; max_new_tokens::Integer=64, temperature::Real=1.0, top_k::Integer=50, rng::AbstractRNG=Random.default_rng(), envelope::Union{Nothing, KVCacheEnvelope}=nothing)
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
        step_logits = logits[:, end, :] ./ Float32(temperature)
        if top_k > 0 && top_k < size(step_logits, 2)
            filtered = copy(step_logits)
            for i in axes(filtered, 1)
                idx = partialsortperm(vec(filtered[i, :]), 1:top_k; rev=true)
                keep = falses(size(filtered, 2))
                keep[idx] .= true
                filtered[i, .!keep] .= -Inf32
            end
            step_logits = filtered
        end
        probs = _softmax_rows(step_logits)
        next_tok = _sample_categorical(probs; rng=rng)
        ids = cat(ids, next_tok; dims=2)
    end
    return ids
end
