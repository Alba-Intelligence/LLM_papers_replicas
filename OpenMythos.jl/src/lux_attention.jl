"""
    LuxGQAttention

Lux-native grouped-query attention matching `GQAttention`.
"""
struct LuxGQAttention <: Lux.LuxCore.AbstractLuxLayer
    dim::Int
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    groups::Int
end

function LuxGQAttention(cfg::MythosConfig)
    head_dim = cfg.dim ÷ cfg.n_heads
    return LuxGQAttention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, head_dim, cfg.n_heads ÷ cfg.n_kv_heads)
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxGQAttention)
    return (
        wq=Float32.(0.02 .* randn(rng, layer.n_heads * layer.head_dim, layer.dim)),
        wk=Float32.(0.02 .* randn(rng, layer.n_kv_heads * layer.head_dim, layer.dim)),
        wv=Float32.(0.02 .* randn(rng, layer.n_kv_heads * layer.head_dim, layer.dim)),
        wo=Float32.(0.02 .* randn(rng, layer.dim, layer.n_heads * layer.head_dim)),
    )
end

Lux.initialstates(::AbstractRNG, ::LuxGQAttention) = NamedTuple()

function (layer::LuxGQAttention)(input::NamedTuple, ps, st)
    x = input.x
    freqs_cis = input.freqs_cis
    mask = get(input, :mask, nothing)
    kv_cache = get(input, :kv_cache, nothing)
    kv_capacity = get(input, :kv_capacity, nothing)
    cache_key = get(input, :cache_key, "default")

    b, t, d = size(x)
    q = reshape(_linear_feature_last(x, ps.wq), b, t, layer.n_heads, layer.head_dim)
    k = reshape(_linear_feature_last(x, ps.wk), b, t, layer.n_kv_heads, layer.head_dim)
    v = reshape(_linear_feature_last(x, ps.wv), b, t, layer.n_kv_heads, layer.head_dim)

    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    if kv_cache !== nothing
        entry = if haskey(kv_cache, cache_key)
            prev = _gq_cache_entry(kv_cache[cache_key])
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

    if layer.groups > 1
        k = repeat(k, 1, 1, layer.groups, 1)
        v = repeat(v, 1, 1, layer.groups, 1)
    end

    qh = permutedims(q, (1, 3, 2, 4))
    kh = permutedims(k, (1, 3, 2, 4))
    vh = permutedims(v, (1, 3, 2, 4))
    out = _attention(qh, kh, vh, mask)
    out = permutedims(out, (1, 3, 2, 4))
    out = reshape(out, b, t, :)
    return _linear_feature_last(out, ps.wo), st
end

_gq_parameters(attn::GQAttention) = (wq=copy(attn.wq), wk=copy(attn.wk), wv=copy(attn.wv), wo=copy(attn.wo))

"""
    LuxMLAttention

Lux-native multi-head latent attention matching `MLAttention`.
"""
struct LuxMLAttention{QN,KN} <: Lux.LuxCore.AbstractLuxLayer
    dim::Int
    n_heads::Int
    q_rank::Int
    kv_lora_rank::Int
    qk_rope_dim::Int
    qk_nope_dim::Int
    v_dim::Int
    q_head_dim::Int
    q_norm::QN
    kv_norm::KN
end

function LuxMLAttention(cfg::MythosConfig)
    return LuxMLAttention(
        cfg.dim,
        cfg.n_heads,
        cfg.q_lora_rank,
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        RMSNorm(cfg.q_lora_rank),
        RMSNorm(cfg.kv_lora_rank),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxMLAttention)
    return (
        q_down=Float32.(0.02 .* randn(rng, layer.q_rank, layer.dim)),
        q_norm=Lux.initialparameters(rng, layer.q_norm),
        q_up_nope=Float32.(0.02 .* randn(rng, layer.n_heads * layer.qk_nope_dim, layer.q_rank)),
        q_up_rope=Float32.(0.02 .* randn(rng, layer.n_heads * layer.qk_rope_dim, layer.q_rank)),
        kv_down=Float32.(0.02 .* randn(rng, layer.kv_lora_rank + layer.qk_rope_dim, layer.dim)),
        kv_norm=Lux.initialparameters(rng, layer.kv_norm),
        kv_up=Float32.(0.02 .* randn(rng, layer.n_heads * (layer.qk_nope_dim + layer.v_dim), layer.kv_lora_rank)),
        wo=Float32.(0.02 .* randn(rng, layer.dim, layer.n_heads * layer.v_dim)),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::LuxMLAttention)
    return (
        q_norm=Lux.initialstates(rng, layer.q_norm),
        kv_norm=Lux.initialstates(rng, layer.kv_norm),
    )
end

function (layer::LuxMLAttention)(input::NamedTuple, ps, st)
    x = input.x
    freqs_cis = input.freqs_cis
    mask = get(input, :mask, nothing)
    kv_cache = get(input, :kv_cache, nothing)
    kv_capacity = get(input, :kv_capacity, nothing)
    cache_key = get(input, :cache_key, "default")

    b, t, d = size(x)

    c_q, st_q = Lux.apply(layer.q_norm, _linear_feature_last(x, ps.q_down), ps.q_norm, st.q_norm)
    q_nope = reshape(_linear_feature_last(c_q, ps.q_up_nope), b, t, layer.n_heads, layer.qk_nope_dim)
    q_rope = reshape(_linear_feature_last(c_q, ps.q_up_rope), b, t, layer.n_heads, layer.qk_rope_dim)
    q_rope = apply_rope(q_rope, freqs_cis)
    q = cat(q_nope, q_rope; dims=4)

    kv_raw = _linear_feature_last(x, ps.kv_down)
    c_kv = kv_raw[:, :, 1:layer.kv_lora_rank]
    k_rope_raw = kv_raw[:, :, (layer.kv_lora_rank + 1):end]
    k_rope = repeat(reshape(k_rope_raw, b, t, 1, layer.qk_rope_dim), 1, 1, layer.n_heads, 1)
    k_rope = apply_rope(k_rope, freqs_cis)

    if kv_cache !== nothing
        entry = if haskey(kv_cache, cache_key)
            prev = _mla_cache_entry(kv_cache[cache_key])
            _reserve_buffer_capacity!(prev.c_kv, kv_capacity)
            _reserve_buffer_capacity!(prev.k_rope, kv_capacity)
            append_axis_buffer!(prev.c_kv, c_kv)
            append_axis_buffer!(prev.k_rope, k_rope)
            prev
        else
            (
                c_kv=filled_axis_buffer(c_kv; axis=2, capacity=_buffer_capacity_hint(kv_capacity, size(c_kv, 2))),
                k_rope=filled_axis_buffer(k_rope; axis=2, capacity=_buffer_capacity_hint(kv_capacity, size(k_rope, 2))),
            )
        end
        kv_cache[cache_key] = entry
        c_kv = buffer_view(entry.c_kv)
        k_rope = buffer_view(entry.k_rope)
    end

    s = size(c_kv, 2)
    kv_normed, st_kv = Lux.apply(layer.kv_norm, c_kv, ps.kv_norm, st.kv_norm)
    kv = reshape(_linear_feature_last(kv_normed, ps.kv_up), b, s, layer.n_heads, layer.qk_nope_dim + layer.v_dim)
    k_nope = kv[:, :, :, 1:layer.qk_nope_dim]
    v = kv[:, :, :, (layer.qk_nope_dim + 1):end]
    k = cat(k_nope, k_rope; dims=4)

    qh = permutedims(q, (1, 3, 2, 4))
    kh = permutedims(k, (1, 3, 2, 4))
    vh = permutedims(v, (1, 3, 2, 4))
    out = _attention(qh, kh, vh, mask)
    out = permutedims(out, (1, 3, 2, 4))
    out = reshape(out, b, t, :)
    return _linear_feature_last(out, ps.wo), (q_norm=st_q, kv_norm=st_kv)
end

_mla_parameters(attn::MLAttention) = (
    q_down=copy(attn.q_down),
    q_norm=(weight=copy(attn.q_norm.weight),),
    q_up_nope=copy(attn.q_up_nope),
    q_up_rope=copy(attn.q_up_rope),
    kv_down=copy(attn.kv_down),
    kv_norm=(weight=copy(attn.kv_norm.weight),),
    kv_up=copy(attn.kv_up),
    wo=copy(attn.wo),
)

to_lux_parameters(attn::GQAttention) = _gq_parameters(attn)
to_lux_parameters(attn::MLAttention) = _mla_parameters(attn)
