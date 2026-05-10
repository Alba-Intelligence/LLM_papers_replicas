"""
    OLMoConfig

Configuration for the OLMo decoder-only transformer family.

The initial Julia slice targets OLMo 2 and therefore centers on dense decoder
blocks with RoPE, QK-Norm, SwiGLU, and OLMo's inside-residual post-norm layout.
"""
Base.@kwdef struct OLMoConfig
    vocab_size::Int = 100_352
    dim::Int = 4_096
    n_heads::Int = 32
    n_kv_heads::Int = 32
    max_seq_len::Int = 4_096
    n_layers::Int = 32
    intermediate_dim::Int = 11_008
    rope_theta::Float32 = 500_000.0f0
    rms_norm_eps::Float32 = 1f-6
    attention_bias::Bool = false
    tie_word_embeddings::Bool = false
    init_scale::Float32 = 0.02f0
end

function _validate_olmo_cfg(cfg::OLMoConfig)
    cfg.vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    cfg.dim > 0 || throw(ArgumentError("dim must be positive"))
    cfg.n_heads > 0 || throw(ArgumentError("n_heads must be positive"))
    cfg.n_kv_heads > 0 || throw(ArgumentError("n_kv_heads must be positive"))
    cfg.n_layers > 0 || throw(ArgumentError("n_layers must be positive"))
    cfg.max_seq_len > 0 || throw(ArgumentError("max_seq_len must be positive"))
    cfg.intermediate_dim > 0 || throw(ArgumentError("intermediate_dim must be positive"))
    cfg.dim % cfg.n_heads == 0 || throw(ArgumentError("dim must be divisible by n_heads"))
    cfg.n_heads % cfg.n_kv_heads == 0 || throw(ArgumentError("n_heads must be divisible by n_kv_heads"))
    cfg.attention_bias && throw(ArgumentError("the current OLMo slice does not implement attention biases yet"))
    cfg.tie_word_embeddings && throw(ArgumentError("the current OLMo slice does not implement tied word embeddings yet"))
    return cfg
end
