function _attention(q::AbstractArray, k::AbstractArray, v::AbstractArray, mask)
    b, h, t, d = size(q)
    dv = size(v, 4)
    mask2 = mask === nothing ? nothing : (ndims(mask) == 4 ? @view(mask[1, 1, :, :]) : mask)
    scale = inv(sqrt(Float32(d)))
    heads = [
        begin
            qmat = @view q[bi, hi, :, :]
            kmat = @view k[bi, hi, :, :]
            scores = (qmat * transpose(kmat)) .* scale
            scores = mask2 === nothing ? scores : scores .+ mask2
            probs = _softmax_rows(scores)
            probs * (@view v[bi, hi, :, :])
        end for bi in 1:b, hi in 1:h
    ]
    batches = [
        cat([reshape(heads[bi, hi], 1, t, dv) for hi in 1:h]...; dims=1)
        for bi in 1:b
    ]
    return cat([reshape(batch, 1, size(batch, 1), size(batch, 2), size(batch, 3)) for batch in batches]...; dims=1)
end

_buffer_capacity_hint(kv_capacity::Union{Nothing, Integer}, chunk_len::Integer) =
    kv_capacity === nothing ? nothing : max(Int(kv_capacity), Int(chunk_len))

function _reserve_buffer_capacity!(buffer, kv_capacity::Union{Nothing, Integer})
    kv_capacity === nothing || ensure_axis_capacity!(buffer, Int(kv_capacity))
    return buffer
end

function _olmo_cache_entry(prev)
    if prev isa NamedTuple{(:k, :v)}
        return prev
    elseif prev isa AbstractDict
        return (k=filled_axis_buffer(prev["k"]; axis=2), v=filled_axis_buffer(prev["v"]; axis=2))
    end
    throw(ArgumentError("unsupported OLMo cache entry type $(typeof(prev))"))
end

"""
    OLMoAttention{T}

OLMo-family causal self-attention with RoPE and QK-Norm.
"""
struct OLMoAttention{T<:AbstractFloat}
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    groups::Int
    q_proj::Matrix{T}
    k_proj::Matrix{T}
    v_proj::Matrix{T}
    o_proj::Matrix{T}
    q_norm::RMSNorm{T}
    k_norm::RMSNorm{T}
end

function OLMoAttention(cfg::OLMoConfig; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    _validate_olmo_cfg(cfg)
    head_dim = cfg.dim ÷ cfg.n_heads
    return OLMoAttention(
        cfg.n_heads,
        cfg.n_kv_heads,
        head_dim,
        cfg.n_heads ÷ cfg.n_kv_heads,
        T.(cfg.init_scale .* randn(rng, cfg.n_heads * head_dim, cfg.dim)),
        T.(cfg.init_scale .* randn(rng, cfg.n_kv_heads * head_dim, cfg.dim)),
        T.(cfg.init_scale .* randn(rng, cfg.n_kv_heads * head_dim, cfg.dim)),
        T.(cfg.init_scale .* randn(rng, cfg.dim, cfg.n_heads * head_dim)),
        RMSNorm(cfg.n_heads * head_dim; eps=cfg.rms_norm_eps, T=T),
        RMSNorm(cfg.n_kv_heads * head_dim; eps=cfg.rms_norm_eps, T=T),
    )
end

"""
    attn(x, freqs_cis; mask=nothing, kv_cache=nothing, kv_capacity=nothing, cache_key="default")

Apply OLMo-family causal self-attention to feature-last hidden states.
"""
function (attn::OLMoAttention)(x::AbstractArray{T, 3}, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, kv_capacity::Union{Nothing, Integer}=nothing, cache_key::String="default") where {T<:AbstractFloat}
    b, t, _ = size(x)

    q = attn.q_norm(_linear_feature_last(x, attn.q_proj))
    k = attn.k_norm(_linear_feature_last(x, attn.k_proj))
    v = _linear_feature_last(x, attn.v_proj)

    q = reshape(q, b, t, attn.n_heads, attn.head_dim)
    k = reshape(k, b, t, attn.n_kv_heads, attn.head_dim)
    v = reshape(v, b, t, attn.n_kv_heads, attn.head_dim)

    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    if kv_cache !== nothing
        entry = if haskey(kv_cache, cache_key)
            prev = _olmo_cache_entry(kv_cache[cache_key])
            _reserve_buffer_capacity!(prev.k, kv_capacity)
            _reserve_buffer_capacity!(prev.v, kv_capacity)
            append_axis_buffer!(prev.k, k)
            append_axis_buffer!(prev.v, v)
            prev
        else
            (
                k=filled_axis_buffer(k; axis=2, capacity=_buffer_capacity_hint(kv_capacity, size(k, 2))),
                v=filled_axis_buffer(v; axis=2, capacity=_buffer_capacity_hint(kv_capacity, size(v, 2))),
            )
        end
        kv_cache[cache_key] = entry
        k = buffer_view(entry.k)
        v = buffer_view(entry.v)
    end

    if attn.groups > 1
        k = repeat(k, 1, 1, attn.groups, 1)
        v = repeat(v, 1, 1, attn.groups, 1)
    end

    qh = permutedims(q, (1, 3, 2, 4))
    kh = permutedims(k, (1, 3, 2, 4))
    vh = permutedims(v, (1, 3, 2, 4))
    out = _attention(qh, kh, vh, mask)
    out = permutedims(out, (1, 3, 2, 4))
    out = reshape(out, b, t, :)
    return _linear_feature_last(out, attn.o_proj)
end
