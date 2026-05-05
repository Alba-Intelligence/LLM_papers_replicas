"""
    build_engram_token_lookup(token_texts)

Build a compressed token-id lookup table for Engram-style hashing by grouping
tokens with the same normalized surface form.

The current normalization is intentionally lightweight and dependency-free:

- lowercase,
- collapse repeated whitespace,
- strip outer whitespace,
- map empty results to a single space sentinel.
"""
function build_engram_token_lookup(token_texts::AbstractVector{<:AbstractString})
    lookup = Vector{Int}(undef, length(token_texts))
    compressed = Dict{String, Int}()
    next_id = 0
    for (idx, text) in enumerate(token_texts)
        key = lowercase(strip(replace(text, r"\s+" => " ")))
        key = isempty(key) ? " " : key
        compressed_id = get!(compressed, key) do
            current = next_id
            next_id += 1
            current
        end
        lookup[idx] = compressed_id
    end
    return lookup
end

function _validate_engram_cfg(cfg::DeepSeekV4Config)
    isempty(cfg.engram_layer_ids) && return cfg
    cfg.engram_max_ngram_size >= 2 || throw(ArgumentError("Engram requires engram_max_ngram_size >= 2"))
    cfg.engram_embed_dim > 0 || throw(ArgumentError("Engram requires engram_embed_dim > 0"))
    cfg.engram_heads_per_ngram > 0 || throw(ArgumentError("Engram requires engram_heads_per_ngram > 0"))
    cfg.engram_embed_dim % cfg.engram_heads_per_ngram == 0 || throw(ArgumentError("Engram requires engram_embed_dim divisible by engram_heads_per_ngram"))
    cfg.engram_vocab_multiplier > 0 || throw(ArgumentError("Engram requires engram_vocab_multiplier > 0"))
    cfg.engram_kernel_size > 0 || throw(ArgumentError("Engram requires engram_kernel_size > 0"))
    all(layer_id -> 1 <= layer_id <= cfg.n_layers, cfg.engram_layer_ids) || throw(ArgumentError("Engram layer ids must fall within 1:n_layers"))
    if cfg.engram_token_lookup !== nothing
        length(cfg.engram_token_lookup) == cfg.vocab_size || throw(ArgumentError("Engram token lookup must match vocab_size"))
        all(id -> id >= 0, cfg.engram_token_lookup) || throw(ArgumentError("Engram token lookup ids must be non-negative"))
    end
    return cfg
end

function _engram_compressed_vocab_size(cfg::DeepSeekV4Config)
    return cfg.engram_token_lookup === nothing ? cfg.vocab_size : maximum(cfg.engram_token_lookup) + 1
end

function _engram_shift_tokens(token_ids::AbstractMatrix{<:Integer}, offset::Integer, pad_id::Integer)
    b, t = size(token_ids)
    if offset == 0
        return Int.(token_ids)
    end
    offset < t || return fill(Int(pad_id), b, t)
    left = fill(Int(pad_id), b, Int(offset))
    right = Int.(token_ids[:, 1:(t - Int(offset))])
    return hcat(left, right)
end

struct EngramHashMapping
    layer_index::Int
    max_ngram_size::Int
    heads_per_ngram::Int
    pad_id::Int
    token_lookup::Union{Nothing, Vector{Int}}
    multipliers::Vector{Int}
    head_vocab_sizes::Vector{Int}
end

function EngramHashMapping(cfg::DeepSeekV4Config, layer_index::Integer)
    _validate_engram_cfg(cfg)
    compressed_vocab_size = _engram_compressed_vocab_size(cfg)
    base_vocab = max(3, compressed_vocab_size * cfg.engram_vocab_multiplier)
    rng = MersenneTwister(cfg.engram_seed + 1_009 * Int(layer_index))
    multipliers = [2 * rand(rng, 1:base_vocab) + 1 for _ in 1:cfg.engram_max_ngram_size]
    total_heads = (cfg.engram_max_ngram_size - 1) * cfg.engram_heads_per_ngram
    head_vocab_sizes = [base_vocab + 2 * (head_idx - 1) + 1 for head_idx in 1:total_heads]
    pad_id = cfg.engram_token_lookup === nothing ? cfg.engram_pad_id : cfg.engram_token_lookup[cfg.engram_pad_id + 1]
    return EngramHashMapping(
        Int(layer_index),
        cfg.engram_max_ngram_size,
        cfg.engram_heads_per_ngram,
        Int(pad_id),
        cfg.engram_token_lookup === nothing ? nothing : copy(cfg.engram_token_lookup),
        multipliers,
        head_vocab_sizes,
    )
end

function (mapping::EngramHashMapping)(token_ids::AbstractMatrix{<:Integer})
    tokens = if mapping.token_lookup === nothing
        Int.(token_ids)
    else
        map(token_id -> mapping.token_lookup[Int(token_id) + 1], token_ids)
    end
    shifts = [_engram_shift_tokens(tokens, offset, mapping.pad_id) for offset in 0:(mapping.max_ngram_size - 1)]
    outputs = Matrix{Int}[]
    for ngram_size in 2:mapping.max_ngram_size
        mix = shifts[1] .* mapping.multipliers[1]
        for idx in 2:ngram_size
            mix = xor.(mix, shifts[idx] .* mapping.multipliers[idx])
        end
        for head_idx in 1:mapping.heads_per_ngram
            vocab_size = mapping.head_vocab_sizes[(ngram_size - 2) * mapping.heads_per_ngram + head_idx]
            push!(outputs, mod.(mix, vocab_size))
        end
    end
    return cat([reshape(out, size(tokens, 1), size(tokens, 2), 1) for out in outputs]...; dims=3)
end

_engram_hash(mapping::EngramHashMapping, token_ids::AbstractMatrix{<:Integer}) = mapping(token_ids)
Zygote.@adjoint _engram_hash(mapping::EngramHashMapping, token_ids::AbstractMatrix{<:Integer}) = _engram_hash(mapping, token_ids), _ -> (nothing, nothing)

"""
    MultiHeadEmbedding{T}

Embedding table for Engram hashed n-gram addresses across all n-gram heads.
"""
struct MultiHeadEmbedding{T<:AbstractFloat}
    offsets::Vector{Int}
    embedding::Matrix{T}
end

function MultiHeadEmbedding(head_vocab_sizes::AbstractVector{<:Integer}, head_dim::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    offsets = cumsum(vcat(0, Int.(head_vocab_sizes[1:(end - 1)])))
    return MultiHeadEmbedding{T}(offsets, T.(0.02 .* randn(rng, sum(Int, head_vocab_sizes), Int(head_dim))))
end

function (embed::MultiHeadEmbedding{T})(input_ids::AbstractArray{<:Integer, 3}) where {T<:AbstractFloat}
    b, t, h = size(input_ids)
    h == length(embed.offsets) || throw(DimensionMismatch("Engram head count mismatch"))
    head_dim = size(embed.embedding, 2)
    outputs = [
        reshape(embed.embedding[input_ids[:, :, head_idx] .+ embed.offsets[head_idx] .+ 1, :], b, t, 1, head_dim)
        for head_idx in 1:h
    ]
    return cat(outputs...; dims=3)
end

struct EngramShortConv{T<:AbstractFloat}
    kernel::Array{T, 3}
end

function EngramShortConv(n_streams::Integer, hidden_size::Integer, kernel_size::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    return EngramShortConv(T.(0.02 .* randn(rng, Int(kernel_size), Int(n_streams), Int(hidden_size))))
end

function _engram_shift_hidden(x::AbstractArray{T, 4}, offset::Integer) where {T<:AbstractFloat}
    b, t, s, d = size(x)
    offset <= 0 && return x
    offset >= t && return zeros(T, b, t, s, d)
    return cat(zeros(T, b, Int(offset), s, d), x[:, 1:(t - Int(offset)), :, :]; dims=2)
end

function (conv::EngramShortConv{T})(x::AbstractArray{T, 4}) where {T<:AbstractFloat}
    b, t, s, d = size(x)
    kernel_size = size(conv.kernel, 1)
    size(conv.kernel, 2) == s || throw(DimensionMismatch("Engram short conv stream count mismatch"))
    size(conv.kernel, 3) == d || throw(DimensionMismatch("Engram short conv hidden size mismatch"))
    return foldl(
        (acc, offset) -> begin
            shifted = _engram_shift_hidden(x, offset - 1)
            kernel = reshape(@view(conv.kernel[offset, :, :]), 1, 1, s, d)
            acc .+ shifted .* kernel
        end,
        1:kernel_size;
        init=zeros(T, b, t, s, d),
    )
end

function _engram_gate_activation(scores::AbstractArray{T}) where {T<:AbstractFloat}
    return _sigmoid.(sign.(scores) .* sqrt.(abs.(scores) .+ eps(T)))
end

"""
    Engram{T}

Optional conditional-memory branch that hashes local n-grams, retrieves static
embeddings, and fuses them into the DeepSeek multi-stream residual state.
"""
struct Engram{T<:AbstractFloat}
    layer_index::Int
    n_streams::Int
    hidden_size::Int
    hash_mapping::EngramHashMapping
    embedding::MultiHeadEmbedding{T}
    short_conv::EngramShortConv{T}
    value_proj::Matrix{T}
    key_projs::Vector{Matrix{T}}
    key_norms::Vector{RMSNorm{T}}
    query_norms::Vector{RMSNorm{T}}
end

function Engram(cfg::DeepSeekV4Config, layer_index::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    _validate_engram_cfg(cfg)
    hash_mapping = EngramHashMapping(cfg, layer_index)
    head_dim = cfg.engram_embed_dim ÷ cfg.engram_heads_per_ngram
    total_embed_dim = (cfg.engram_max_ngram_size - 1) * cfg.engram_embed_dim
    n_streams = cfg.n_hyper_connections
    hidden_size = cfg.dim
    return Engram(
        Int(layer_index),
        n_streams,
        hidden_size,
        hash_mapping,
        MultiHeadEmbedding(hash_mapping.head_vocab_sizes, head_dim; rng=rng, T=T),
        EngramShortConv(n_streams, hidden_size, cfg.engram_kernel_size; rng=rng, T=T),
        T.(0.02 .* randn(rng, hidden_size, total_embed_dim)),
        [T.(0.02 .* randn(rng, hidden_size, total_embed_dim)) for _ in 1:n_streams],
        [RMSNorm(hidden_size; T=T) for _ in 1:n_streams],
        [RMSNorm(hidden_size; T=T) for _ in 1:n_streams],
    )
end

function (engram::Engram{T})(hidden_states::AbstractArray{T, 4}, token_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    b, t, s, d = size(hidden_states)
    s == engram.n_streams || throw(DimensionMismatch("Engram stream count mismatch"))
    d == engram.hidden_size || throw(DimensionMismatch("Engram hidden size mismatch"))

    hashed = _engram_hash(engram.hash_mapping, token_ids)
    embeddings = engram.embedding(hashed)
    flat_embeddings = reshape(embeddings, b, t, :)
    value = reshape(_linear_feature_last(flat_embeddings, engram.value_proj), b, t, 1, d)
    gates = cat(
        [
            begin
                key = _linear_feature_last(flat_embeddings, engram.key_projs[stream_idx])
                normed_key = engram.key_norms[stream_idx](key)
                query = engram.query_norms[stream_idx](@view hidden_states[:, :, stream_idx, :])
                raw_gate = sum(normed_key .* query; dims=3) ./ T(sqrt(Float32(d)))
                reshape(_engram_gate_activation(raw_gate), b, t, 1, 1)
            end
            for stream_idx in 1:s
        ]...;
        dims=3,
    )
    gated = gates .* value
    return gated .+ engram.short_conv(gated)
end
