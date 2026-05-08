module OpenMythos

using Base64
using LinearAlgebra
using Random
using Serialization
using Statistics
import Lux
import Optimisers
import Zygote
import TransformerCore
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
include("ffn.jl")
include("blocks.jl")
include("recurrent.jl")
include("model.jl")
include("variants.jl")
include("lux_layers.jl")
include("lux_attention.jl")
include("lux_blocks.jl")
include("lux_model.jl")
include("tokenizer.jl")
include("training.jl")

export MythosConfig,
       RMSNorm,
       precompute_rope_freqs,
       apply_rope,
       loop_index_embedding,
       GQAttention,
       MLAttention,
       Expert,
       MoEFFN,
       TransformerBlock,
       RecurrentBlock,
       LoRAAdapter,
       LTIInjection,
       get_A,
       ACTHalting,
       LuxExpert,
       LuxMoEFFN,
       LuxLoRAAdapter,
       LuxLTIInjection,
       LuxACTHalting,
       LuxGQAttention,
       LuxMLAttention,
       LuxTransformerBlock,
       LuxRecurrentBlock,
       LuxOpenMythos,
       to_lux_parameters,
       OpenMythos,
       generate,
       DEFAULT_MODEL_ID,
       MythosTokenizer,
       vocab_size,
       tokenize,
       detokenize,
       WarmupCosineSchedule,
       learning_rate,
       bootstrap_training_config,
       bootstrap_full_model_training_config,
       chunk_next_token_pairs,
       text_next_token_pairs,
       batch_next_token_pairs,
       fineweb_edu_batches,
       KVCacheEnvelope,
       save_kv_cache,
       load_kv_cache,
       LuxHeadOnlyOpenMythos,
       HeadOnlyTrainerState,
       FullModelTrainerState,
       LuxFullModelTrainerState,
       head_only_logits,
       head_only_loss,
       full_model_logits,
       full_model_loss,
       lux_full_model_logits,
       lux_full_model_loss,
       train_head_only_step!,
       train_full_model_step!,
       train_lux_full_model_step!,
       train_head_only!,
       train_full_model!,
       train_lux_full_model!,
       chunked_prefill,
       latest_checkpoint,
       save_head_only_checkpoint,
       load_head_only_checkpoint,
       save_full_model_checkpoint,
       load_full_model_checkpoint,
       save_lux_full_model_checkpoint,
       load_lux_full_model_checkpoint,
       mythos_1b,
       mythos_3b,
       mythos_10b,
       mythos_50b,
       mythos_100b,
       mythos_500b,
       mythos_1t

end
