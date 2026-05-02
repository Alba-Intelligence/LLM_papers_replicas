module DeepSeekV4

using LinearAlgebra
using Random
using Serialization
using Statistics
import Lux
import Optimisers
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
                       apply_rope,
                       WarmupCosineSchedule,
                       learning_rate,
                       chunk_next_token_pairs,
                       text_next_token_pairs,
                       batch_next_token_pairs,
                       latest_checkpoint,
                       _head_loss_and_grad,
                       AxisAppendBuffer,
                       filled_axis_buffer,
                       ensure_axis_capacity!,
                       append_axis_buffer!,
                       buffer_view,
                       KVCacheEnvelope,
                       reserve_kv_capacity!,
                       save_kv_cache,
                       load_kv_cache

include("config.jl")
include("ffn.jl")
include("attention.jl")
include("mhc.jl")
include("deepseek_model.jl")
include("variants.jl")
include("training.jl")

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
       WarmupCosineSchedule,
       learning_rate,
       chunk_next_token_pairs,
       text_next_token_pairs,
       batch_next_token_pairs,
       latest_checkpoint,
       KVCacheEnvelope,
       save_kv_cache,
       load_kv_cache,
       bootstrap_deepseek_training_config,
       LuxHeadOnlyDeepSeekV4,
       DeepSeekHeadTrainerState,
       deepseek_head_logits,
       deepseek_head_loss,
       train_deepseek_head_only_step!,
       train_deepseek_head_only!,
       save_deepseek_checkpoint,
       load_deepseek_checkpoint,
       chunked_prefill,
       deepseek_v4_tiny,
       deepseek_v4_flash,
       deepseek_v4_pro

end
