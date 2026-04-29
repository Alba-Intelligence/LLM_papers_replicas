function _attention(q::AbstractArray, k::AbstractArray, v::AbstractArray, mask)
    b, h, t, d = size(q)
    s = size(k, 3)
    dv = size(v, 4)
    out = zeros(promote_type(eltype(q), eltype(k), eltype(v)), b, h, t, dv)
    mask2 = mask === nothing ? nothing : (ndims(mask) == 4 ? @view(mask[1, 1, :, :]) : mask)
    scale = inv(sqrt(Float32(d)))

    for bi in 1:b, hi in 1:h
        qmat = Array(@view q[bi, hi, :, :])
        kmat = Array(@view k[bi, hi, :, :])
        scores = (qmat * transpose(kmat)) .* scale
        if mask2 !== nothing
            scores .+= mask2
        end
        probs = _softmax_rows(scores)
        vmat = Array(@view v[bi, hi, :, :])
        @views out[bi, hi, :, :] .= probs * vmat
    end

    return out
end

struct GQAttention{T<:AbstractFloat}
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    groups::Int
    wq::Matrix{T}
    wk::Matrix{T}
    wv::Matrix{T}
    wo::Matrix{T}
end

function GQAttention(cfg::MythosConfig; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    head_dim = cfg.dim ÷ cfg.n_heads
    return GQAttention(
        cfg.n_heads,
        cfg.n_kv_heads,
        head_dim,
        cfg.n_heads ÷ cfg.n_kv_heads,
        T.(0.02 .* randn(rng, cfg.n_heads * head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.n_kv_heads * head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.n_kv_heads * head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.dim, cfg.n_heads * head_dim)),
    )
end

function (attn::GQAttention)(x::AbstractArray, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, cache_key::String="default")
    b, t, _ = size(x)
    q = reshape(_linear_feature_last(x, attn.wq), b, t, attn.n_heads, attn.head_dim)
    k = reshape(_linear_feature_last(x, attn.wk), b, t, attn.n_kv_heads, attn.head_dim)
    v = reshape(_linear_feature_last(x, attn.wv), b, t, attn.n_kv_heads, attn.head_dim)

    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    if kv_cache !== nothing
        if haskey(kv_cache, cache_key)
            prev = kv_cache[cache_key]
            k = cat(prev["k"], k; dims=2)
            v = cat(prev["v"], v; dims=2)
        end
        kv_cache[cache_key] = Dict("k" => copy(k), "v" => copy(v))
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
    return _linear_feature_last(out, attn.wo)
end

struct MLAttention{T<:AbstractFloat}
    n_heads::Int
    kv_lora_rank::Int
    qk_rope_dim::Int
    qk_nope_dim::Int
    v_dim::Int
    q_head_dim::Int
    q_down::Matrix{T}
    q_norm::RMSNorm{T}
    q_up_nope::Matrix{T}
    q_up_rope::Matrix{T}
    kv_down::Matrix{T}
    kv_norm::RMSNorm{T}
    kv_up::Matrix{T}
    wo::Matrix{T}
end

function MLAttention(cfg::MythosConfig; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    return MLAttention(
        cfg.n_heads,
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        T.(0.02 .* randn(rng, cfg.q_lora_rank, cfg.dim)),
        RMSNorm(cfg.q_lora_rank; T=T),
        T.(0.02 .* randn(rng, cfg.n_heads * cfg.qk_nope_head_dim, cfg.q_lora_rank)),
        T.(0.02 .* randn(rng, cfg.n_heads * cfg.qk_rope_head_dim, cfg.q_lora_rank)),
        T.(0.02 .* randn(rng, cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.dim)),
        RMSNorm(cfg.kv_lora_rank; T=T),
        T.(0.02 .* randn(rng, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank)),
        T.(0.02 .* randn(rng, cfg.dim, cfg.n_heads * cfg.v_head_dim)),
    )
end

function (attn::MLAttention)(x::AbstractArray, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, cache_key::String="default")
    b, t, _ = size(x)

    c_q = attn.q_norm(_linear_feature_last(x, attn.q_down))
    q_nope = reshape(_linear_feature_last(c_q, attn.q_up_nope), b, t, attn.n_heads, attn.qk_nope_dim)
    q_rope = reshape(_linear_feature_last(c_q, attn.q_up_rope), b, t, attn.n_heads, attn.qk_rope_dim)
    q_rope = apply_rope(q_rope, freqs_cis)
    q = cat(q_nope, q_rope; dims=4)

    kv_raw = _linear_feature_last(x, attn.kv_down)
    c_kv = kv_raw[:, :, 1:attn.kv_lora_rank]
    k_rope_raw = kv_raw[:, :, (attn.kv_lora_rank + 1):end]
    k_rope = repeat(reshape(k_rope_raw, b, t, 1, attn.qk_rope_dim), 1, 1, attn.n_heads, 1)
    k_rope = apply_rope(k_rope, freqs_cis)

    if kv_cache !== nothing
        if haskey(kv_cache, cache_key)
            prev = kv_cache[cache_key]
            c_kv = cat(prev["c_kv"], c_kv; dims=2)
            k_rope = cat(prev["k_rope"], k_rope; dims=2)
        end
        kv_cache[cache_key] = Dict("c_kv" => copy(c_kv), "k_rope" => copy(k_rope))
    end

    s = size(c_kv, 2)
    kv = reshape(_linear_feature_last(attn.kv_norm(c_kv), attn.kv_up), b, s, attn.n_heads, attn.qk_nope_dim + attn.v_dim)
    k_nope = kv[:, :, :, 1:attn.qk_nope_dim]
    v = kv[:, :, :, (attn.qk_nope_dim + 1):end]
    k = cat(k_nope, k_rope; dims=4)

    qh = permutedims(q, (1, 3, 2, 4))
    kh = permutedims(k, (1, 3, 2, 4))
    vh = permutedims(v, (1, 3, 2, 4))
    out = _attention(qh, kh, vh, mask)
    out = permutedims(out, (1, 3, 2, 4))
    out = reshape(out, b, t, :)
    return _linear_feature_last(out, attn.wo)
end
