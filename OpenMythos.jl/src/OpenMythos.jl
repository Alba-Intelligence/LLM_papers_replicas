module OpenMythos

using Base64
using LinearAlgebra
using Random
using Serialization
using Statistics
import Lux
import NNlib
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
                       apply_rope

function _causal_mask(seq_len::Integer, ::Type{T}=Float32) where {T<:AbstractFloat}
    mask = zeros(T, 1, 1, seq_len, seq_len)
    for i in 1:seq_len, j in (i + 1):seq_len
        mask[1, 1, i, j] = T(-Inf)
    end
    return mask
end

include("config.jl")
include("attention.jl")
include("ffn.jl")
include("blocks.jl")
include("recurrent.jl")
include("model.jl")
include("variants.jl")
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
       chunk_next_token_pairs,
       text_next_token_pairs,
       batch_next_token_pairs,
       fineweb_edu_batches,
       LuxHeadOnlyOpenMythos,
       HeadOnlyTrainerState,
       head_only_logits,
       head_only_loss,
       train_head_only_step!,
       train_head_only!,
       latest_checkpoint,
       save_head_only_checkpoint,
       load_head_only_checkpoint,
       mythos_1b,
       mythos_3b,
       mythos_10b,
       mythos_50b,
       mythos_100b,
       mythos_500b,
       mythos_1t

end
