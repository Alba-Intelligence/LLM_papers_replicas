"""
    DeepSeekV4

Julia package for the architecture-first DeepSeek V4 implementation in this
workspace.

The package currently focuses on the released attention, MoE, mHC, MTP, runtime,
and bootstrap-training surfaces while sharing common tensor/runtime helpers with
`TransformerCore.jl`.
"""
module DeepSeekV4

using LinearAlgebra
using Random
using Serialization
using Statistics
import Lux
import Optimisers
import Zygote
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
include("engram.jl")
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
       Engram,
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
       build_engram_token_lookup,
        bootstrap_deepseek_training_config,
        bootstrap_deepseek_full_model_training_config,
        LuxHeadOnlyDeepSeekV4,
        DeepSeekHeadTrainerState,
        DeepSeekFullModelTrainerState,
        deepseek_head_logits,
        deepseek_head_loss,
        deepseek_full_model_logits,
        deepseek_full_model_loss,
        train_deepseek_head_only_step!,
        train_deepseek_full_model_step!,
        train_deepseek_head_only!,
        train_deepseek_full_model!,
        save_deepseek_checkpoint,
        load_deepseek_checkpoint,
        save_deepseek_full_model_checkpoint,
        load_deepseek_full_model_checkpoint,
        chunked_prefill,
        deepseek_v4_tiny,
        deepseek_v4_tiny_engram,
        deepseek_v4_flash,
       deepseek_v4_pro

end
