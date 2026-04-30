function _rope_suffix_dim(head_dim::Integer)
    return max(0, min(64, Int(head_dim) - Int(head_dim) % 2))
end

function _apply_partial_rope(x::AbstractArray{T, 4}, freqs_cis::AbstractMatrix{<:Complex}, rope_dim::Integer; inverse::Bool=false) where {T<:AbstractFloat}
    rope_dim == 0 && return x
    d = size(x, 4)
    rope_dim <= d || throw(DimensionMismatch("RoPE suffix exceeds head dimension"))
    start_dim = d - rope_dim + 1
    out = copy(x)
    rope_freqs = inverse ? conj.(freqs_cis[:, 1:(rope_dim ÷ 2)]) : freqs_cis[:, 1:(rope_dim ÷ 2)]
    @views out[:, :, :, start_dim:end] .= apply_rope(x[:, :, :, start_dim:end], rope_freqs)
    return out
end

function _compress_csa_entries(
    ca::AbstractMatrix{T},
    cb::AbstractMatrix{T},
    za::AbstractMatrix{T},
    zb::AbstractMatrix{T},
    bias_a::AbstractMatrix{T},
    bias_b::AbstractMatrix{T},
    compression::Integer,
) where {T<:AbstractFloat}
    n, d = size(ca)
    n == size(cb, 1) == size(za, 1) == size(zb, 1) || throw(DimensionMismatch("CSA inputs must share sequence length"))
    d == size(cb, 2) == size(za, 2) == size(zb, 2) || throw(DimensionMismatch("CSA inputs must share width"))
    compression > 0 || throw(ArgumentError("compression must be positive"))

    nblocks = cld(n, compression)
    compressed = zeros(T, nblocks, d)
    for block in 1:nblocks
        start_idx = (block - 1) * compression + 1
        stop_idx = min(block * compression, n)
        prev_start = max(1, start_idx - compression)
        prev_stop = start_idx - 1
        len_a = stop_idx - start_idx + 1
        len_b = max(0, prev_stop - prev_start + 1)
        for dim_idx in 1:d
            weights = T[]
            values = T[]
            for offset in 1:len_a
                push!(weights, za[start_idx + offset - 1, dim_idx] + bias_a[offset, dim_idx])
                push!(values, ca[start_idx + offset - 1, dim_idx])
            end
            for offset in 1:len_b
                push!(weights, zb[prev_start + offset - 1, dim_idx] + bias_b[offset, dim_idx])
                push!(values, cb[prev_start + offset - 1, dim_idx])
            end
            isempty(weights) && continue
            m = maximum(weights)
            probs = exp.(weights .- m)
            probs ./= sum(probs)
            compressed[block, dim_idx] = sum(probs .* values)
        end
    end
    return compressed
end

function _compress_hca_entries(
    c::AbstractMatrix{T},
    z::AbstractMatrix{T},
    bias::AbstractMatrix{T},
    compression::Integer,
) where {T<:AbstractFloat}
    n, d = size(c)
    size(z) == (n, d) || throw(DimensionMismatch("HCA inputs must share shape"))
    compression > 0 || throw(ArgumentError("compression must be positive"))

    nblocks = cld(n, compression)
    compressed = zeros(T, nblocks, d)
    for block in 1:nblocks
        start_idx = (block - 1) * compression + 1
        stop_idx = min(block * compression, n)
        len_block = stop_idx - start_idx + 1
        for dim_idx in 1:d
            weights = [z[start_idx + offset - 1, dim_idx] + bias[offset, dim_idx] for offset in 1:len_block]
            values = [c[start_idx + offset - 1, dim_idx] for offset in 1:len_block]
            m = maximum(weights)
            probs = exp.(weights .- m)
            probs ./= sum(probs)
            compressed[block, dim_idx] = sum(probs .* values)
        end
    end
    return compressed
end

function _attention_with_sink(query::AbstractVector{T}, keys::AbstractMatrix{T}, values::AbstractMatrix{T}, sink_logit::T) where {T<:AbstractFloat}
    isempty(keys) && return zeros(T, size(values, 2))
    logits = (keys * query) ./ T(sqrt(Float32(length(query))))
    m = max(maximum(logits), sink_logit)
    exp_logits = exp.(logits .- m)
    denom = sum(exp_logits) + exp(sink_logit - m)
    probs = exp_logits ./ denom
    return vec(transpose(probs) * values)
end

function _grouped_output_projection(outputs::AbstractArray{T, 4}, group_projs::Vector{Matrix{T}}, wo::AbstractMatrix{T}) where {T<:AbstractFloat}
    b, t, h, d = size(outputs)
    ngroups = length(group_projs)
    h % ngroups == 0 || throw(DimensionMismatch("number of heads must be divisible by output groups"))
    per_group = h ÷ ngroups
    intermediates = Vector{Array{T, 3}}(undef, ngroups)
    for group_idx in 1:ngroups
        head_range = ((group_idx - 1) * per_group + 1):(group_idx * per_group)
        chunk = reshape(outputs[:, :, head_range, :], b, t, :)
        intermediates[group_idx] = _linear_feature_last(chunk, group_projs[group_idx])
    end
    merged = cat(intermediates...; dims=3)
    return _linear_feature_last(merged, wo)
end

function _eligible_compressed_blocks(abs_pos::Int, compression::Int, nblocks::Int)
    abs_pos <= 1 && return 0
    return clamp(fld(abs_pos - 2, compression), 0, nblocks)
end

function _csa_cache_entry(prev)
    if prev isa NamedTuple{(:ca, :cb, :za, :zb, :kia, :kib)}
        return prev
    elseif prev isa AbstractDict
        return (
            ca=filled_axis_buffer(prev["ca"]; axis=2),
            cb=filled_axis_buffer(prev["cb"]; axis=2),
            za=filled_axis_buffer(prev["za"]; axis=2),
            zb=filled_axis_buffer(prev["zb"]; axis=2),
            kia=filled_axis_buffer(prev["kia"]; axis=2),
            kib=filled_axis_buffer(prev["kib"]; axis=2),
        )
    end
    throw(ArgumentError("unsupported CSA cache entry type $(typeof(prev))"))
end

function _hca_cache_entry(prev)
    if prev isa NamedTuple{(:c, :z)}
        return prev
    elseif prev isa AbstractDict
        return (c=filled_axis_buffer(prev["c"]; axis=2), z=filled_axis_buffer(prev["z"]; axis=2))
    end
    throw(ArgumentError("unsupported HCA cache entry type $(typeof(prev))"))
end

struct CompressedSparseAttention{T<:AbstractFloat}
    n_heads::Int
    head_dim::Int
    rope_dim::Int
    compression::Int
    topk::Int
    window::Int
    indexer_heads::Int
    indexer_dim::Int
    q_down::Matrix{T}
    q_up::Matrix{T}
    q_index_up::Matrix{T}
    q_index_weight::Matrix{T}
    kv_a::Matrix{T}
    kv_b::Matrix{T}
    z_a::Matrix{T}
    z_b::Matrix{T}
    indexer_k_a::Matrix{T}
    indexer_k_b::Matrix{T}
    bias_a::Matrix{T}
    bias_b::Matrix{T}
    q_norm::RMSNorm{T}
    kv_norm::RMSNorm{T}
    sink_logits::Vector{T}
    group_projs::Vector{Matrix{T}}
    wo::Matrix{T}
end

function CompressedSparseAttention(cfg::DeepSeekV4Config; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    head_dim = cfg.dim ÷ cfg.n_heads
    cfg.dim % cfg.n_heads == 0 || throw(DimensionMismatch("DeepSeekV4 dim must be divisible by n_heads"))
    cfg.n_heads % cfg.attention_groups == 0 || throw(DimensionMismatch("n_heads must be divisible by attention_groups"))
    cfg.indexer_dim <= head_dim || throw(DimensionMismatch("indexer_dim must be <= attention head_dim"))
    per_group = cfg.n_heads ÷ cfg.attention_groups
    return CompressedSparseAttention(
        cfg.n_heads,
        head_dim,
        _rope_suffix_dim(head_dim),
        cfg.csa_compression,
        cfg.attention_topk,
        cfg.attention_window,
        cfg.indexer_heads,
        cfg.indexer_dim,
        T.(0.02 .* randn(rng, cfg.query_compression_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.n_heads * head_dim, cfg.query_compression_dim)),
        T.(0.02 .* randn(rng, cfg.indexer_heads * cfg.indexer_dim, cfg.query_compression_dim)),
        T.(0.02 .* randn(rng, cfg.indexer_heads, cfg.dim)),
        T.(0.02 .* randn(rng, head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.indexer_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.indexer_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.csa_compression, head_dim)),
        T.(0.02 .* randn(rng, cfg.csa_compression, head_dim)),
        RMSNorm(head_dim; T=T),
        RMSNorm(head_dim; T=T),
        zeros(T, cfg.n_heads),
        [T.(0.02 .* randn(rng, cfg.attention_group_dim, per_group * head_dim)) for _ in 1:cfg.attention_groups],
        T.(0.02 .* randn(rng, cfg.dim, cfg.attention_groups * cfg.attention_group_dim)),
    )
end

function (attn::CompressedSparseAttention)(x::AbstractArray{T, 3}, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, cache_key::String="default", start_pos::Integer=0) where {T<:AbstractFloat}
    b, t, _ = size(x)
    c_q = _linear_feature_last(x, attn.q_down)
    q = reshape(_linear_feature_last(c_q, attn.q_up), b, t, attn.n_heads, attn.head_dim)
    q_index = reshape(_linear_feature_last(c_q, attn.q_index_up), b, t, attn.indexer_heads, attn.indexer_dim)
    q = attn.q_norm(q)
    q = _apply_partial_rope(q, freqs_cis, attn.rope_dim)

    ca_cur = _linear_feature_last(x, attn.kv_a)
    cb_cur = _linear_feature_last(x, attn.kv_b)
    za_cur = _linear_feature_last(x, attn.z_a)
    zb_cur = _linear_feature_last(x, attn.z_b)
    kia_cur = _linear_feature_last(x, attn.indexer_k_a)
    kib_cur = _linear_feature_last(x, attn.indexer_k_b)

    ca_cur = reshape(attn.kv_norm(reshape(ca_cur, b * t, attn.head_dim)), b, t, attn.head_dim)
    cb_cur = reshape(attn.kv_norm(reshape(cb_cur, b * t, attn.head_dim)), b, t, attn.head_dim)
    idx_norm = RMSNorm(attn.indexer_dim; T=T)
    kia_cur = reshape(idx_norm(reshape(kia_cur, b * t, attn.indexer_dim)), b, t, attn.indexer_dim)
    kib_cur = reshape(idx_norm(reshape(kib_cur, b * t, attn.indexer_dim)), b, t, attn.indexer_dim)

    ca4 = reshape(ca_cur, b, t, 1, attn.head_dim)
    cb4 = reshape(cb_cur, b, t, 1, attn.head_dim)
    ca_cur = reshape(_apply_partial_rope(ca4, freqs_cis, attn.rope_dim), b, t, attn.head_dim)
    cb_cur = reshape(_apply_partial_rope(cb4, freqs_cis, attn.rope_dim), b, t, attn.head_dim)

    ca = ca_cur
    cb = cb_cur
    za = za_cur
    zb = zb_cur
    kia = kia_cur
    kib = kib_cur
    if kv_cache !== nothing
        entry = if haskey(kv_cache, cache_key)
            prev = _csa_cache_entry(kv_cache[cache_key])
            append_axis_buffer!(prev.ca, ca_cur)
            append_axis_buffer!(prev.cb, cb_cur)
            append_axis_buffer!(prev.za, za_cur)
            append_axis_buffer!(prev.zb, zb_cur)
            append_axis_buffer!(prev.kia, kia_cur)
            append_axis_buffer!(prev.kib, kib_cur)
            prev
        else
            (
                ca=filled_axis_buffer(ca_cur; axis=2),
                cb=filled_axis_buffer(cb_cur; axis=2),
                za=filled_axis_buffer(za_cur; axis=2),
                zb=filled_axis_buffer(zb_cur; axis=2),
                kia=filled_axis_buffer(kia_cur; axis=2),
                kib=filled_axis_buffer(kib_cur; axis=2),
            )
        end
        kv_cache[cache_key] = entry
        ca = buffer_view(entry.ca)
        cb = buffer_view(entry.cb)
        za = buffer_view(entry.za)
        zb = buffer_view(entry.zb)
        kia = buffer_view(entry.kia)
        kib = buffer_view(entry.kib)
    end

    out_heads = zeros(T, b, t, attn.n_heads, attn.head_dim)
    indexer_weights = _linear_feature_last(x, attn.q_index_weight)
    for bi in 1:b
        ca_seq = Array(@view ca[bi, :, :])
        cb_seq = Array(@view cb[bi, :, :])
        za_seq = Array(@view za[bi, :, :])
        zb_seq = Array(@view zb[bi, :, :])
        compressed = _compress_csa_entries(ca_seq, cb_seq, za_seq, zb_seq, attn.bias_a, attn.bias_b, attn.compression)
        indexer_comp = _compress_csa_entries(
            Array(@view kia[bi, :, :]),
            Array(@view kib[bi, :, :]),
            za_seq[:, 1:attn.indexer_dim],
            zb_seq[:, 1:attn.indexer_dim],
            attn.bias_a[:, 1:attn.indexer_dim],
            attn.bias_b[:, 1:attn.indexer_dim],
            attn.compression,
        )
        nblocks = size(compressed, 1)
        for ti in 1:t
            abs_pos = start_pos + ti
            eligible = _eligible_compressed_blocks(abs_pos, attn.compression, nblocks)
            local_start = max(1, abs_pos - attn.window)
            local_stop = max(0, abs_pos - 1)
            local_values = local_stop >= local_start ? Array(@view ca_seq[local_start:local_stop, :]) : zeros(T, 0, attn.head_dim)

            selected = Int[]
            if eligible > 0
                scores = zeros(T, eligible)
                for block_idx in 1:eligible
                    for ih in 1:attn.indexer_heads
                        q_idx = vec(@view q_index[bi, ti, ih, :])
                        scores[block_idx] += indexer_weights[bi, ti, ih] * max(zero(T), dot(q_idx, @view indexer_comp[block_idx, :]))
                    end
                end
                ksel = min(attn.topk, eligible)
                selected = partialsortperm(scores, 1:ksel; rev=true)
            end

            sparse_values = isempty(selected) ? zeros(T, 0, attn.head_dim) : compressed[selected, :]
            keys = vcat(sparse_values, local_values)
            for hi in 1:attn.n_heads
                qvec = vec(@view q[bi, ti, hi, :])
                if size(keys, 1) > 0
                    out_heads[bi, ti, hi, :] .= _attention_with_sink(qvec, keys, keys, attn.sink_logits[hi])
                end
            end
        end
    end

    out_heads = _apply_partial_rope(out_heads, freqs_cis, attn.rope_dim; inverse=true)
    return _grouped_output_projection(out_heads, attn.group_projs, attn.wo)
end

struct HeavilyCompressedAttention{T<:AbstractFloat}
    n_heads::Int
    head_dim::Int
    rope_dim::Int
    compression::Int
    window::Int
    q_down::Matrix{T}
    q_up::Matrix{T}
    kv::Matrix{T}
    z::Matrix{T}
    bias::Matrix{T}
    q_norm::RMSNorm{T}
    kv_norm::RMSNorm{T}
    sink_logits::Vector{T}
    group_projs::Vector{Matrix{T}}
    wo::Matrix{T}
end

function HeavilyCompressedAttention(cfg::DeepSeekV4Config; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    head_dim = cfg.dim ÷ cfg.n_heads
    cfg.dim % cfg.n_heads == 0 || throw(DimensionMismatch("DeepSeekV4 dim must be divisible by n_heads"))
    cfg.n_heads % cfg.attention_groups == 0 || throw(DimensionMismatch("n_heads must be divisible by attention_groups"))
    per_group = cfg.n_heads ÷ cfg.attention_groups
    return HeavilyCompressedAttention(
        cfg.n_heads,
        head_dim,
        _rope_suffix_dim(head_dim),
        cfg.hca_compression,
        cfg.attention_window,
        T.(0.02 .* randn(rng, cfg.query_compression_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.n_heads * head_dim, cfg.query_compression_dim)),
        T.(0.02 .* randn(rng, head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, head_dim, cfg.dim)),
        T.(0.02 .* randn(rng, cfg.hca_compression, head_dim)),
        RMSNorm(head_dim; T=T),
        RMSNorm(head_dim; T=T),
        zeros(T, cfg.n_heads),
        [T.(0.02 .* randn(rng, cfg.attention_group_dim, per_group * head_dim)) for _ in 1:cfg.attention_groups],
        T.(0.02 .* randn(rng, cfg.dim, cfg.attention_groups * cfg.attention_group_dim)),
    )
end

function (attn::HeavilyCompressedAttention)(x::AbstractArray{T, 3}, freqs_cis::AbstractMatrix; mask=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing, cache_key::String="default", start_pos::Integer=0) where {T<:AbstractFloat}
    b, t, _ = size(x)
    c_q = _linear_feature_last(x, attn.q_down)
    q = reshape(_linear_feature_last(c_q, attn.q_up), b, t, attn.n_heads, attn.head_dim)
    q = attn.q_norm(q)
    q = _apply_partial_rope(q, freqs_cis, attn.rope_dim)

    c_cur = _linear_feature_last(x, attn.kv)
    z_cur = _linear_feature_last(x, attn.z)
    c_cur = reshape(attn.kv_norm(reshape(c_cur, b * t, attn.head_dim)), b, t, attn.head_dim)
    c4 = reshape(c_cur, b, t, 1, attn.head_dim)
    c_cur = reshape(_apply_partial_rope(c4, freqs_cis, attn.rope_dim), b, t, attn.head_dim)

    c = c_cur
    z = z_cur
    if kv_cache !== nothing
        entry = if haskey(kv_cache, cache_key)
            prev = _hca_cache_entry(kv_cache[cache_key])
            append_axis_buffer!(prev.c, c_cur)
            append_axis_buffer!(prev.z, z_cur)
            prev
        else
            (c=filled_axis_buffer(c_cur; axis=2), z=filled_axis_buffer(z_cur; axis=2))
        end
        kv_cache[cache_key] = entry
        c = buffer_view(entry.c)
        z = buffer_view(entry.z)
    end

    out_heads = zeros(T, b, t, attn.n_heads, attn.head_dim)
    for bi in 1:b
        c_seq = Array(@view c[bi, :, :])
        z_seq = Array(@view z[bi, :, :])
        compressed = _compress_hca_entries(c_seq, z_seq, attn.bias, attn.compression)
        nblocks = size(compressed, 1)
        for ti in 1:t
            abs_pos = start_pos + ti
            eligible = _eligible_compressed_blocks(abs_pos, attn.compression, nblocks)
            local_start = max(1, abs_pos - attn.window)
            local_stop = max(0, abs_pos - 1)
            local_values = local_stop >= local_start ? Array(@view c_seq[local_start:local_stop, :]) : zeros(T, 0, attn.head_dim)
            dense_values = eligible > 0 ? compressed[1:eligible, :] : zeros(T, 0, attn.head_dim)
            keys = vcat(dense_values, local_values)
            for hi in 1:attn.n_heads
                qvec = vec(@view q[bi, ti, hi, :])
                if size(keys, 1) > 0
                    out_heads[bi, ti, hi, :] .= _attention_with_sink(qvec, keys, keys, attn.sink_logits[hi])
                end
            end
        end
    end

    out_heads = _apply_partial_rope(out_heads, freqs_cis, attn.rope_dim; inverse=true)
    return _grouped_output_projection(out_heads, attn.group_projs, attn.wo)
end
