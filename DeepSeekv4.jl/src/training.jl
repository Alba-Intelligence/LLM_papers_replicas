"""
    bootstrap_deepseek_training_config(vocab_size; seq_len=64)

Return a small DeepSeek V4 configuration for head-only bootstrap training and
smoke tests.
"""
function bootstrap_deepseek_training_config(vocab_size::Integer; seq_len::Integer=64, with_engram::Bool=false, engram_token_lookup::Union{Nothing, Vector{Int}}=nothing)
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    cfg = DeepSeekV4Config(
        vocab_size=Int(vocab_size),
        dim=64,
        n_heads=4,
        max_seq_len=Int(seq_len),
        n_layers=4,
        csa_compression=4,
        hca_compression=8,
        attention_topk=2,
        attention_window=4,
        query_compression_dim=16,
        indexer_dim=8,
        indexer_heads=2,
        attention_groups=2,
        attention_group_dim=16,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        hash_routed_layers=1,
        mtp_tokens=2,
        n_hyper_connections=2,
        engram_layer_ids=with_engram ? [1, 3] : Int[],
        engram_embed_dim=32,
        engram_heads_per_ngram=2,
        engram_vocab_multiplier=2,
        engram_kernel_size=4,
        engram_token_lookup=engram_token_lookup,
    )
    return cfg
end

"""
    bootstrap_deepseek_full_model_training_config(vocab_size; seq_len=64)

Return a smaller DeepSeek V4 configuration for the current full-model bootstrap
training slice.
"""
function bootstrap_deepseek_full_model_training_config(vocab_size::Integer; seq_len::Integer=64, with_engram::Bool=false, engram_token_lookup::Union{Nothing, Vector{Int}}=nothing)
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    return DeepSeekV4Config(
        vocab_size=Int(vocab_size),
        dim=32,
        n_heads=4,
        max_seq_len=Int(seq_len),
        n_layers=2,
        csa_compression=2,
        hca_compression=2,
        attention_topk=1,
        attention_window=2,
        query_compression_dim=8,
        indexer_dim=4,
        indexer_heads=1,
        attention_groups=2,
        attention_group_dim=8,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=8,
        hash_routed_layers=1,
        mtp_tokens=1,
        n_hyper_connections=2,
        sinkhorn_iters=4,
        engram_layer_ids=with_engram ? [1] : Int[],
        engram_embed_dim=16,
        engram_heads_per_ngram=2,
        engram_vocab_multiplier=2,
        engram_kernel_size=3,
        engram_token_lookup=engram_token_lookup,
    )
end

function _validate_deepseek_full_model_cfg(cfg::DeepSeekV4Config)
    cfg.vocab_size > 0 || throw(ArgumentError("full-model training requires vocab_size > 0"))
    cfg.dim > 0 || throw(ArgumentError("full-model training requires dim > 0"))
    cfg.n_heads > 0 || throw(ArgumentError("full-model training requires n_heads > 0"))
    cfg.n_layers > 0 || throw(ArgumentError("full-model training requires n_layers > 0"))
    cfg.max_seq_len > 0 || throw(ArgumentError("full-model training requires max_seq_len > 0"))
    cfg.n_experts > 0 || throw(ArgumentError("full-model training requires n_experts > 0"))
    cfg.n_shared_experts >= 0 || throw(ArgumentError("full-model training requires n_shared_experts >= 0"))
    cfg.n_experts_per_tok > 0 || throw(ArgumentError("full-model training requires n_experts_per_tok > 0"))
    cfg.n_experts_per_tok <= cfg.n_experts || throw(ArgumentError("full-model training requires n_experts_per_tok <= n_experts"))
    cfg.hash_routed_layers >= 0 || throw(ArgumentError("full-model training requires hash_routed_layers >= 0"))
    cfg.hash_routed_layers <= cfg.n_layers || throw(ArgumentError("full-model training requires hash_routed_layers <= n_layers"))
    return cfg
end

function _sequence_cross_entropy(logits::AbstractArray{T, 3}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    size(logits, 1) == size(target_ids, 1) || throw(DimensionMismatch("batch size mismatch"))
    size(logits, 2) == size(target_ids, 2) || throw(DimensionMismatch("sequence length mismatch"))

    rows = permutedims(_flatten_feature_last(logits), (2, 1))
    flat_targets = vec(target_ids)
    shifted = rows .- maximum(rows; dims=2)
    log_denom = log.(sum(exp.(shifted); dims=2))
    total = zero(T)
    for i in eachindex(flat_targets)
        tgt = flat_targets[i] + 1
        (1 <= tgt <= size(rows, 2)) || throw(BoundsError(rows, (i, tgt)))
        total -= shifted[i, tgt] - log_denom[i]
    end
    return total / T(length(flat_targets))
end

function _mtp_sequence_cross_entropy(logits::AbstractArray{T, 4}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    size(logits, 1) == size(target_ids, 1) || throw(DimensionMismatch("batch size mismatch"))
    size(logits, 2) == size(target_ids, 2) || throw(DimensionMismatch("sequence length mismatch"))

    max_heads = min(size(logits, 3), size(target_ids, 2) - 1)
    max_heads <= 0 && return zero(T)

    total = zero(T)
    for head_idx in 1:max_heads
        valid_t = size(target_ids, 2) - head_idx
        total += _sequence_cross_entropy(
            @view(logits[:, 1:valid_t, head_idx, :]),
            @view(target_ids[:, (head_idx + 1):end]),
        )
    end
    return total / T(max_heads)
end

function _tree_sumsq(x)
    if x === nothing
        return 0.0
    elseif x isa AbstractArray{<:Number}
        return sum(abs2, x)
    elseif x isa Number
        return abs2(x)
    elseif x isa NamedTuple
        return sum(_tree_sumsq, values(x))
    elseif x isa Tuple
        return sum(_tree_sumsq, x)
    elseif x isa AbstractArray
        return sum(_tree_sumsq, x)
    elseif isstructtype(typeof(x))
        return sum(_tree_sumsq(getfield(x, i)) for i in 1:fieldcount(typeof(x)))
    end
    return 0.0
end

function _deepseek_full_model_loss(model::DeepSeekV4Model{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    main_loss = _sequence_cross_entropy(model(input_ids), target_ids)
    if isempty(model.mtp_heads)
        return main_loss
    end
    mtp_loss = _mtp_sequence_cross_entropy(mtp_logits(model, input_ids), target_ids)
    return (main_loss + mtp_loss) / T(2)
end

"""
    LuxHeadOnlyDeepSeekV4

Lux wrapper that exposes only the DeepSeek V4 LM head as trainable parameters.
"""
struct LuxHeadOnlyDeepSeekV4{M<:DeepSeekV4Model} <: Lux.LuxCore.AbstractLuxLayer
    model::M
end

Lux.initialparameters(::AbstractRNG, layer::LuxHeadOnlyDeepSeekV4) = (head=copy(layer.model.head),)
Lux.initialstates(::AbstractRNG, ::LuxHeadOnlyDeepSeekV4) = NamedTuple()

function (layer::LuxHeadOnlyDeepSeekV4)(input_ids::AbstractMatrix{<:Integer}, ps, st)
    hidden = deepseek_hidden(layer.model, input_ids)
    return _linear_feature_last(hidden, ps.head), st
end

"""
    DeepSeekHeadTrainerState{T}

Mutable training state for the Lux-backed DeepSeek V4 head-only bootstrap path.
This now reuses `TransformerCore.NextTokenTrainerState` for optimization,
checkpointing, and the generic next-token training loop.
"""
mutable struct DeepSeekHeadTrainerState{T<:AbstractFloat, TS}
    layer::LuxHeadOnlyDeepSeekV4{DeepSeekV4Model{T}}
    trainer::TS
end

"""
    DeepSeekFullModelTrainerState{T}

Mutable training state for the current DeepSeek V4 full-model bootstrap trainer.
This now reuses `TransformerCore.NextTokenTrainerState` as the common optimizer /
checkpoint holder even though the loss is still DeepSeek-specific because of the
MTP branch.
"""
mutable struct DeepSeekFullModelTrainerState{T<:AbstractFloat, TS}
    trainer::TS
end

function Base.getproperty(state::DeepSeekHeadTrainerState, name::Symbol)
    if name === :head
        return getfield(state, :trainer).ps.head
    elseif name === :lux_state
        return getfield(state, :trainer).st
    elseif name === :opt_state
        return getfield(state, :trainer).opt_state
    elseif name === :schedule
        return getfield(state, :trainer).schedule
    elseif name === :step
        return getfield(state, :trainer).step
    end
    return getfield(state, name)
end

function Base.getproperty(state::DeepSeekFullModelTrainerState, name::Symbol)
    if name === :model
        return getfield(state, :trainer).ps
    elseif name === :opt_state
        return getfield(state, :trainer).opt_state
    elseif name === :schedule
        return getfield(state, :trainer).schedule
    elseif name === :step
        return getfield(state, :trainer).step
    end
    return getfield(state, name)
end

function DeepSeekFullModelTrainerState(
    model::DeepSeekV4Model{T};
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
) where {T<:AbstractFloat}
    _validate_deepseek_full_model_cfg(model.cfg)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(; eta=zero(T), beta=(T(beta1), T(beta2)), lambda=T(weight_decay), epsilon=T(eps)),
        model,
    )
    trainer = NextTokenTrainerState(model, NamedTuple(), opt_state, schedule, 0)
    return DeepSeekFullModelTrainerState{T, typeof(trainer)}(trainer)
end

function DeepSeekHeadTrainerState(
    model::DeepSeekV4Model{T};
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
) where {T<:AbstractFloat}
    layer = LuxHeadOnlyDeepSeekV4(model)
    trainer = NextTokenTrainerState(
        layer;
        schedule=schedule,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    return DeepSeekHeadTrainerState{T, typeof(trainer)}(layer, trainer)
end

"""Return LM logits from the DeepSeek head-only trainer state."""
deepseek_head_logits(state::DeepSeekHeadTrainerState, input_ids::AbstractMatrix{<:Integer}) =
    next_token_logits(state.layer, state.trainer, input_ids)

"""Return sequence cross-entropy for the DeepSeek head-only trainer state."""
deepseek_head_loss(state::DeepSeekHeadTrainerState, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) =
    next_token_loss(state.layer, state.trainer, input_ids, target_ids)

"""Return LM logits from the full-model DeepSeek trainer state."""
function deepseek_full_model_logits(state::DeepSeekFullModelTrainerState, input_ids::AbstractMatrix{<:Integer})
    return state.trainer.ps(input_ids)
end

"""Return sequence cross-entropy for the full-model DeepSeek trainer state."""
function deepseek_full_model_loss(state::DeepSeekFullModelTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    return _deepseek_full_model_loss(state.trainer.ps, input_ids, target_ids)
end

"""Take one optimization step in the DeepSeek V4 head-only trainer."""
function train_deepseek_head_only_step!(state::DeepSeekHeadTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    return train_next_token_step!(state.trainer, state.layer, input_ids, target_ids)
end

"""Take one optimization step in the DeepSeek V4 full-model trainer."""
function train_deepseek_full_model_step!(state::DeepSeekFullModelTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    size(input_ids) == size(target_ids) || throw(ArgumentError("input_ids and target_ids must have the same shape"))
    loss, grads = Zygote.withgradient(model -> _deepseek_full_model_loss(model, input_ids, target_ids), state.trainer.ps)
    grad = only(grads)

    lr = learning_rate(state.trainer.schedule, state.trainer.step)
    Optimisers.adjust!(state.trainer.opt_state, lr)
    state.trainer.opt_state, state.trainer.ps = Optimisers.update(state.trainer.opt_state, state.trainer.ps, grad)
    state.trainer.step += 1

    return (loss=loss, lr=lr, grad_norm=T(sqrt(_tree_sumsq(grad))), step=state.trainer.step)
end

"""Serialize the DeepSeek V4 head-only trainer state to a shared checkpoint root."""
function save_deepseek_checkpoint(state::DeepSeekHeadTrainerState, ckpt_root::AbstractString; keep_last::Integer=3, metadata::AbstractDict=Dict{String, Any}())
    return save_trainer_checkpoint(
        state.trainer,
        ckpt_root;
        family="deepseekv4",
        mode="head_only",
        keep_last=keep_last,
        config=state.layer.model.cfg,
        metadata=metadata,
    )
end

"""Serialize the DeepSeek V4 full-model trainer state to a shared checkpoint root."""
function save_deepseek_full_model_checkpoint(state::DeepSeekFullModelTrainerState, ckpt_root::AbstractString; keep_last::Integer=3, metadata::AbstractDict=Dict{String, Any}())
    return save_trainer_checkpoint(
        state.trainer,
        ckpt_root;
        family="deepseekv4",
        mode="full_model",
        keep_last=keep_last,
        config=state.trainer.ps.cfg,
        metadata=metadata,
    )
end

"""Restore a DeepSeek V4 head-only trainer state from `path` for `model`."""
function load_deepseek_checkpoint(path::AbstractString, model::DeepSeekV4Model{T}) where {T<:AbstractFloat}
    restored = load_trainer_checkpoint(path; expected_family="deepseekv4", expected_mode="head_only")
    size(restored.state.ps.head) == size(model.head) || throw(ArgumentError("checkpoint head shape does not match model"))
    trainer = NextTokenTrainerState(
        (head=T.(restored.state.ps.head),),
        restored.state.st,
        restored.state.opt_state,
        restored.state.schedule,
        restored.state.step,
    )
    return DeepSeekHeadTrainerState{T, typeof(trainer)}(LuxHeadOnlyDeepSeekV4(model), trainer)
end

"""Restore a DeepSeek V4 full-model trainer state from `path`."""
function load_deepseek_full_model_checkpoint(path::AbstractString)
    restored = load_trainer_checkpoint(path; expected_family="deepseekv4", expected_mode="full_model")
    model = restored.state.ps
    _validate_deepseek_full_model_cfg(model.cfg)
    trainer = NextTokenTrainerState(
        model,
        restored.state.st,
        restored.state.opt_state,
        restored.state.schedule,
        restored.state.step,
    )
    return DeepSeekFullModelTrainerState{eltype(model.head), typeof(trainer)}(trainer)
end

"""Run a multi-step DeepSeek V4 head-only training loop over `batches`."""
function train_deepseek_head_only!(
    state::DeepSeekHeadTrainerState,
    batches::AbstractVector{<:Tuple{<:AbstractMatrix{<:Integer}, <:AbstractMatrix{<:Integer}}};
    total_steps::Integer=state.schedule.total_steps,
    log_every::Integer=10,
    ckpt_dir::Union{Nothing, AbstractString}=nothing,
    ckpt_every::Integer=0,
    keep_last::Integer=3,
    io::IO=stdout,
    checkpoint_metadata::AbstractDict=Dict{String, Any}(),
)
    return train_next_token!(
        state.trainer,
        state.layer,
        batches;
        total_steps=total_steps,
        log_every=log_every,
        io=io,
        ckpt_root=ckpt_dir,
        ckpt_every=ckpt_every,
        family="deepseekv4",
        mode="head_only",
        keep_last=keep_last,
        checkpoint_config=state.layer.model.cfg,
        checkpoint_metadata=checkpoint_metadata,
    )
end

"""Run a multi-step DeepSeek V4 full-model training loop over `batches`."""
function train_deepseek_full_model!(
    state::DeepSeekFullModelTrainerState,
    batches::AbstractVector{<:Tuple{<:AbstractMatrix{<:Integer}, <:AbstractMatrix{<:Integer}}};
    total_steps::Integer=state.schedule.total_steps,
    log_every::Integer=10,
    ckpt_dir::Union{Nothing, AbstractString}=nothing,
    ckpt_every::Integer=0,
    keep_last::Integer=3,
    io::IO=stdout,
    checkpoint_metadata::AbstractDict=Dict{String, Any}(),
)
    isempty(batches) && throw(ArgumentError("at least one batch is required"))
    total_steps >= state.step || throw(ArgumentError("total_steps must be >= current step"))

    last_metrics = (loss=zero(eltype(state.model.head)), lr=zero(eltype(state.model.head)), grad_norm=zero(eltype(state.model.head)), step=state.step)
    while state.step < total_steps
        batch = batches[mod1(state.step + 1, length(batches))]
        last_metrics = train_deepseek_full_model_step!(state, batch[1], batch[2])

        if log_every > 0 && (state.step == 1 || state.step % log_every == 0 || state.step == total_steps)
            println(io, "step $(state.step)/$(total_steps) | loss $(round(last_metrics.loss; digits=4)) | gnorm $(round(last_metrics.grad_norm; digits=4)) | lr $(last_metrics.lr)")
        end

        if ckpt_dir !== nothing && ckpt_every > 0 && state.step % ckpt_every == 0
            save_deepseek_full_model_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
        end
    end

    if ckpt_dir !== nothing && ckpt_every > 0 && state.step > 0 && state.step % ckpt_every != 0
        save_deepseek_full_model_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
    end

    return last_metrics
end
