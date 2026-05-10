"""Return the tiny OLMo 2 preset used for tests and smoke runs."""
olmo2_tiny() = OLMoConfig(
    vocab_size=512,
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    max_seq_len=64,
    n_layers=4,
    intermediate_dim=256,
    rope_theta=500_000.0f0,
)

"""Return the published OLMo 2 7B base configuration."""
olmo2_7b() = OLMoConfig(
    vocab_size=100_352,
    dim=4_096,
    n_heads=32,
    n_kv_heads=32,
    max_seq_len=4_096,
    n_layers=32,
    intermediate_dim=11_008,
    rope_theta=500_000.0f0,
)
