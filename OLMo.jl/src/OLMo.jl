"""
    OLMo

Julia package for the OLMo model family in this workspace.

The first slice targets OLMo 2 as a dense decoder-only transformer with MHA,
QK-Norm, RoPE, SwiGLU, and the OLMo-specific inside-residual post-norm block
layout.
"""
module OLMo

using LinearAlgebra
using Random
using Serialization
import Optimisers
import TextDataCore
import TransformerCore
import Zygote
using TransformerCore: _silu,
                       _linear_feature_last,
                       _softmax_rows,
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
                       next_token_cross_entropy,
                       NextTokenTrainerState,
                       latest_checkpoint,
                       save_trainer_checkpoint,
                       load_trainer_checkpoint,
                       filled_axis_buffer,
                       ensure_axis_capacity!,
                       append_axis_buffer!,
                       buffer_view,
                       KVCacheEnvelope,
                       reserve_kv_capacity!,
                       save_kv_cache,
                       load_kv_cache

const DEFAULT_TOKENIZER_MODEL_ID = "cl100k_base"

function _causal_mask(seq_len::Integer, prefix_len::Integer=0, ::Type{T}=Float32) where {T<:AbstractFloat}
    seq_len >= 0 || throw(ArgumentError("seq_len must be non-negative"))
    prefix_len >= 0 || throw(ArgumentError("prefix_len must be non-negative"))
    total_len = Int(prefix_len) + Int(seq_len)
    q_idx = reshape(collect(1:Int(seq_len)), Int(seq_len), 1)
    k_idx = reshape(collect(1:total_len), 1, total_len)
    blocked = k_idx .> (Int(prefix_len) .+ q_idx)
    return reshape(ifelse.(blocked, T(-Inf), zero(T)), 1, 1, Int(seq_len), total_len)
end

include("config.jl")
include("attention.jl")
include("blocks.jl")
include("model.jl")
include("variants.jl")
include("tokenizer.jl")
include("training.jl")

export DEFAULT_TOKENIZER_MODEL_ID,
       OLMoConfig,
       RMSNorm,
       precompute_rope_freqs,
       apply_rope,
       OLMoAttention,
       OLMoMLP,
       OLMoBlock,
       OLMoModel,
       olmo_hidden,
       generate,
       chunked_prefill,
       OLMoTokenizer,
       vocab_size,
       tokenize,
       detokenize,
       encode,
       decode,
       WarmupCosineSchedule,
       learning_rate,
       chunk_next_token_pairs,
       text_next_token_pairs,
       batch_next_token_pairs,
       latest_checkpoint,
       KVCacheEnvelope,
       save_kv_cache,
       load_kv_cache,
       bootstrap_olmo_training_config,
       OLMoFullModelTrainerState,
       olmo_logits,
       olmo_loss,
       train_olmo_step!,
       train_olmo!,
       save_olmo_checkpoint,
       load_olmo_checkpoint,
       olmo2_tiny,
       olmo2_7b

end
