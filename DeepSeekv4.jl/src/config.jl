"""
    DeepSeekV4Config

Configuration for the DeepSeek V4 architecture-first Julia model.

The fields cover compressed-attention parameters, grouped output projection,
Mixture-of-Experts routing, multi-token prediction heads, manifold hyper
connections, and long-context runtime limits.
"""
Base.@kwdef struct DeepSeekV4Config
    vocab_size::Int = 102_400
    dim::Int = 2_048
    n_heads::Int = 16
    max_seq_len::Int = 4_096
    n_layers::Int = 24
    rope_theta::Float32 = 10_000.0f0
    csa_compression::Int = 8
    hca_compression::Int = 32
    attention_topk::Int = 8
    attention_window::Int = 128
    query_compression_dim::Int = 512
    indexer_dim::Int = 64
    indexer_heads::Int = 4
    attention_groups::Int = 4
    attention_group_dim::Int = 256
    n_experts::Int = 64
    n_shared_experts::Int = 1
    n_experts_per_tok::Int = 6
    expert_dim::Int = 1_408
    hash_routed_layers::Int = 2
    mtp_tokens::Int = 2
    n_hyper_connections::Int = 2
    sinkhorn_iters::Int = 20
    attention_pattern::NTuple{2, Symbol} = (:csa, :hca)
    dropout::Float32 = 0.0f0
    engram_layer_ids::Vector{Int} = Int[]
    engram_max_ngram_size::Int = 3
    engram_embed_dim::Int = 256
    engram_heads_per_ngram::Int = 4
    engram_vocab_multiplier::Int = 4
    engram_kernel_size::Int = 4
    engram_seed::Int = 0
    engram_pad_id::Int = 0
    engram_token_lookup::Union{Nothing, Vector{Int}} = nothing
end
