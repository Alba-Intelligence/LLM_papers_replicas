abstract type AbstractModelConfig end

"""
    MythosConfig

Configuration for the OpenMythos recurrent-depth transformer.

The fields cover model width, attention backend selection, recurrent depth,
Mixture-of-Experts settings, LoRA/ACT parameters, and long-context runtime
limits.
"""
Base.@kwdef struct MythosConfig <: AbstractModelConfig
    vocab_size::Int = 32_000
    dim::Int = 2_048
    n_heads::Int = 16
    n_kv_heads::Int = 4
    max_seq_len::Int = 4_096
    max_loop_iters::Int = 16
    prelude_layers::Int = 2
    coda_layers::Int = 2
    attn_type::String = "mla"
    kv_lora_rank::Int = 512
    q_lora_rank::Int = 1_536
    qk_rope_head_dim::Int = 64
    qk_nope_head_dim::Int = 128
    v_head_dim::Int = 128
    n_experts::Int = 64
    n_shared_experts::Int = 2
    n_experts_per_tok::Int = 4
    expert_dim::Int = 512
    act_threshold::Float32 = 0.99f0
    rope_theta::Float32 = 500_000.0f0
    lora_rank::Int = 16
    max_output_tokens::Int = 4_096
    dropout::Float32 = 0.0f0
end
