"""
    bootstrap_olmo_training_config(vocab_size; seq_len=64)

Return a small OLMo configuration for bootstrap training and smoke tests.
"""
function bootstrap_olmo_training_config(vocab_size::Integer; seq_len::Integer=64)
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    return OLMoConfig(
        vocab_size=Int(vocab_size),
        dim=64,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=Int(seq_len),
        n_layers=4,
        intermediate_dim=256,
        rope_theta=500_000.0f0,
    )
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

"""
    OLMoFullModelTrainerState{T}

Mutable training state for the current OLMo full-model bootstrap trainer.
"""
mutable struct OLMoFullModelTrainerState{T<:AbstractFloat, TS}
    trainer::TS
end

function OLMoFullModelTrainerState(trainer::NextTokenTrainerState{<:OLMoModel{T}}) where {T<:AbstractFloat}
    return OLMoFullModelTrainerState{T, typeof(trainer)}(trainer)
end

function Base.getproperty(state::OLMoFullModelTrainerState, name::Symbol)
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

function Base.setproperty!(state::OLMoFullModelTrainerState, name::Symbol, value)
    if name === :model
        getfield(state, :trainer).ps = value
    elseif name === :opt_state
        getfield(state, :trainer).opt_state = value
    elseif name === :schedule
        getfield(state, :trainer).schedule = value
    elseif name === :step
        getfield(state, :trainer).step = value
    else
        setfield!(state, name, value)
    end
    return value
end

function OLMoFullModelTrainerState(
    model::OLMoModel{T};
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
) where {T<:AbstractFloat}
    _validate_olmo_cfg(model.cfg)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(; eta=zero(T), beta=(T(beta1), T(beta2)), lambda=T(weight_decay), epsilon=T(eps)),
        model,
    )
    trainer = NextTokenTrainerState(model, NamedTuple(), opt_state, schedule, 0)
    return OLMoFullModelTrainerState(trainer)
end

"""Return LM logits from an OLMo trainer state."""
olmo_logits(state::OLMoFullModelTrainerState, input_ids::AbstractMatrix{<:Integer}) = state.model(input_ids)

"""Return next-token loss from an OLMo trainer state."""
olmo_loss(state::OLMoFullModelTrainerState, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) =
    next_token_cross_entropy(olmo_logits(state, input_ids), target_ids)

"""
    train_olmo_step!(state, input_ids, target_ids)

Take one optimization step for the current OLMo full-model trainer.
"""
function train_olmo_step!(state::OLMoFullModelTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    size(input_ids) == size(target_ids) || throw(ArgumentError("input_ids and target_ids must have the same shape"))

    loss_ref = Ref(zero(T))
    grad = Zygote.gradient(state.model) do model
        loss = next_token_cross_entropy(model(input_ids), target_ids)
        loss_ref[] = loss
        return loss
    end[1]

    lr = learning_rate(state.schedule, state.step)
    Optimisers.adjust!(state.opt_state, lr)
    state.opt_state, state.model = Optimisers.update(state.opt_state, state.model, grad)
    state.step += 1

    return (loss=loss_ref[], lr=lr, grad_norm=T(sqrt(_tree_sumsq(grad))), step=state.step)
end

"""
    train_olmo!(state, batches; ...)

Run a multi-step OLMo next-token training loop over `(input_ids, target_ids)`
batches.
"""
function train_olmo!(
    state::OLMoFullModelTrainerState,
    batches::AbstractVector{<:Tuple{<:AbstractMatrix{<:Integer}, <:AbstractMatrix{<:Integer}}};
    total_steps::Integer=state.schedule.total_steps,
    log_every::Integer=10,
    ckpt_root::Union{Nothing, AbstractString}=nothing,
    ckpt_every::Integer=0,
    keep_last::Integer=3,
    io::IO=stdout,
    checkpoint_metadata::AbstractDict=Dict{String, Any}(),
)
    isempty(batches) && throw(ArgumentError("at least one batch is required"))
    total_steps >= state.step || throw(ArgumentError("total_steps must be >= current step"))

    loss_type = typeof(state.schedule.max_lr)
    last_metrics = (loss=zero(loss_type), lr=zero(loss_type), grad_norm=zero(loss_type), step=state.step)
    while state.step < total_steps
        batch = batches[mod1(state.step + 1, length(batches))]
        last_metrics = train_olmo_step!(state, batch[1], batch[2])

        if log_every > 0 && (state.step == 1 || state.step % log_every == 0 || state.step == total_steps)
            println(io, "step $(state.step)/$(total_steps) | loss $(round(last_metrics.loss; digits=4)) | gnorm $(round(last_metrics.grad_norm; digits=4)) | lr $(last_metrics.lr)")
        end

        if ckpt_root !== nothing && ckpt_every > 0 && state.step % ckpt_every == 0
            save_olmo_checkpoint(state, ckpt_root; keep_last=keep_last, metadata=checkpoint_metadata)
        end
    end

    if ckpt_root !== nothing && ckpt_every > 0 && state.step > 0 && state.step % ckpt_every != 0
        save_olmo_checkpoint(state, ckpt_root; keep_last=keep_last, metadata=checkpoint_metadata)
    end

    return last_metrics
end

"""
    save_olmo_checkpoint(state, root; keep_last=3, metadata=Dict())

Serialize an OLMo full-model trainer checkpoint using the shared
family/mode-aware checkpoint layout.
"""
function save_olmo_checkpoint(
    state::OLMoFullModelTrainerState,
    root::AbstractString;
    keep_last::Integer=3,
    metadata::AbstractDict=Dict{String, Any}(),
)
    return save_trainer_checkpoint(
        state.trainer,
        root;
        family="olmo",
        mode="full_model",
        keep_last=keep_last,
        config=state.model.cfg,
        metadata=metadata,
    )
end

"""
    load_olmo_checkpoint(path)

Load an OLMo full-model trainer checkpoint.
"""
function load_olmo_checkpoint(path::AbstractString)
    restored = load_trainer_checkpoint(path; expected_family="olmo", expected_mode="full_model")
    return OLMoFullModelTrainerState(restored.state)
end
