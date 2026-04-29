module DeepSeekV4

using LinearAlgebra
using Random
using Statistics
using TransformerCore: _sigmoid,
                       _silu,
                       _flatten_feature_last,
                       _unflatten_feature_last,
                       _feature_last_shape,
                       _feature_broadcast,
                       _linear_feature_last,
                       _softmax_rows,
                       _softmax_cols,
                       _embed_tokens,
                       _sample_categorical,
                       RMSNorm,
                       precompute_rope_freqs,
                       apply_rope

include("config.jl")
include("ffn.jl")
include("attention.jl")
include("mhc.jl")
include("deepseek_model.jl")
include("variants.jl")

export DeepSeekV4Config,
       RMSNorm,
       precompute_rope_freqs,
       apply_rope,
       Expert,
       MoEFFN,
       HashMoEFFN,
       CompressedSparseAttention,
       HeavilyCompressedAttention,
       ManifoldHyperConnections,
       DeepSeekV4Block,
       DeepSeekV4Model,
       deepseek_hidden,
       mtp_logits,
       generate,
       deepseek_v4_tiny,
       deepseek_v4_flash,
       deepseek_v4_pro

end
