mutable struct NextTokenTrainerState{PS,ST,OS,T<:AbstractFloat}
    ps::PS
    st::ST
    opt_state::OS
    schedule::WarmupCosineSchedule{T}
    step::Int
end

function NextTokenTrainerState(
    layer;
    rng::AbstractRNG=Random.default_rng(),
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
) where {T<:AbstractFloat}
    ps, st = Lux.setup(rng, layer)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(; eta=zero(T), beta=(T(beta1), T(beta2)), lambda=T(weight_decay), epsilon=T(eps)),
        ps,
    )
    return NextTokenTrainerState(ps, st, opt_state, schedule, 0)
end

function _tree_map(f::Function, x)
    if x === nothing
        return nothing
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}((_tree_map(f, value) for value in values(x)))
    else
        return f(x)
    end
end

function _tree_zip_map(f::Function, a, b)
    if a === nothing || b === nothing
        return nothing
    elseif a isa NamedTuple && b isa NamedTuple
        keys(a) == keys(b) || throw(ArgumentError("tree key mismatch"))
        return NamedTuple{keys(a)}((_tree_zip_map(f, av, bv) for (av, bv) in zip(values(a), values(b))))
    else
        return f(a, b)
    end
end

function _zero_tree_like(x)
    if x === nothing
        return nothing
    elseif x isa AbstractArray{<:Number}
        return zero.(x)
    elseif x isa Number
        return zero(x)
    elseif x isa AbstractArray
        return map(_zero_tree_like, x)
    end
    return zero(x)
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
    end
    return 0.0
end

"""
    parameter_mask(tree, predicate)

Build a boolean tree matching the `NamedTuple` structure of `tree`. `predicate(path, leaf)`
receives a tuple path of field names and the leaf value.
"""
function parameter_mask(tree, predicate::Function; path::Tuple=())
    if tree === nothing
        return nothing
    elseif tree isa NamedTuple
        return NamedTuple{keys(tree)}((parameter_mask(value, predicate; path=(path..., key)) for (key, value) in pairs(tree)))
    else
        return Bool(predicate(path, tree))
    end
end

"""
    apply_gradient_mask(grad, mask)

Zero out gradient leaves where the corresponding boolean `mask` leaf is `false`.
"""
function apply_gradient_mask(grad, mask)
    return _tree_zip_map(
        (g, keep) -> keep ? g : _zero_tree_like(g),
        grad,
        mask,
    )
end

"""Return logits from a next-token language model layer."""
function next_token_logits(layer, state::NextTokenTrainerState, input_ids::AbstractMatrix{<:Integer})
    logits, _ = Lux.apply(layer, input_ids, state.ps, state.st)
    return logits
end

"""Return next-token cross-entropy from a trainer state."""
function next_token_loss(layer, state::NextTokenTrainerState, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer})
    logits = next_token_logits(layer, state, input_ids)
    return next_token_cross_entropy(logits, target_ids)
end

"""
    train_next_token_step!(state, layer, input_ids, target_ids; grad_mask=nothing)

Take one optimization step for a Lux-native next-token language model.
"""
function train_next_token_step!(
    state::NextTokenTrainerState{PS,ST,OS,T},
    layer,
    input_ids::AbstractMatrix{<:Integer},
    target_ids::AbstractMatrix{<:Integer};
    grad_mask=nothing,
) where {PS,ST,OS,T<:AbstractFloat}
    size(input_ids) == size(target_ids) || throw(ArgumentError("input_ids and target_ids must have the same shape"))

    st_ref = Ref(state.st)
    loss_ref = Ref(zero(T))
    grad = Zygote.gradient(state.ps) do ps
        logits, st_new = Lux.apply(layer, input_ids, ps, state.st)
        st_ref[] = st_new
        loss = next_token_cross_entropy(logits, target_ids)
        loss_ref[] = loss
        return loss
    end[1]

    grad = grad_mask === nothing ? grad : apply_gradient_mask(grad, grad_mask)
    lr = learning_rate(state.schedule, state.step)
    Optimisers.adjust!(state.opt_state, lr)
    state.opt_state, state.ps = Optimisers.update(state.opt_state, state.ps, grad)
    state.st = st_ref[]
    state.step += 1

    return (loss=loss_ref[], lr=lr, grad_norm=T(sqrt(_tree_sumsq(grad))), step=state.step)
end

"""
    train_next_token!(state, layer, batches; ...)

Run a multi-step next-token training loop over `(input_ids, target_ids)` batches.
"""
function train_next_token!(
    state::NextTokenTrainerState,
    layer,
    batches::AbstractVector{<:Tuple{<:AbstractMatrix{<:Integer}, <:AbstractMatrix{<:Integer}}};
    total_steps::Integer=state.schedule.total_steps,
    log_every::Integer=10,
    ckpt_root::Union{Nothing, AbstractString}=nothing,
    family::Union{Nothing, AbstractString}=nothing,
    mode::Union{Nothing, AbstractString}=nothing,
    ckpt_every::Integer=0,
    keep_last::Integer=3,
    io::IO=stdout,
    checkpoint_config=nothing,
    checkpoint_metadata::AbstractDict=Dict{String, Any}(),
    grad_mask=nothing,
)
    isempty(batches) && throw(ArgumentError("at least one batch is required"))
    total_steps >= state.step || throw(ArgumentError("total_steps must be >= current step"))
    if ckpt_root !== nothing
        family === nothing && throw(ArgumentError("family is required when ckpt_root is set"))
        mode === nothing && throw(ArgumentError("mode is required when ckpt_root is set"))
    end

    loss_type = typeof(state.schedule.max_lr)
    last_metrics = (loss=zero(loss_type), lr=zero(loss_type), grad_norm=zero(loss_type), step=state.step)
    while state.step < total_steps
        batch = batches[mod1(state.step + 1, length(batches))]
        last_metrics = train_next_token_step!(state, layer, batch[1], batch[2]; grad_mask=grad_mask)

        if log_every > 0 && (state.step == 1 || state.step % log_every == 0 || state.step == total_steps)
            println(io, "step $(state.step)/$(total_steps) | loss $(round(last_metrics.loss; digits=4)) | gnorm $(round(last_metrics.grad_norm; digits=4)) | lr $(last_metrics.lr)")
        end

        if ckpt_root !== nothing && ckpt_every > 0 && state.step % ckpt_every == 0
            save_trainer_checkpoint(
                state,
                ckpt_root;
                family=family,
                mode=mode,
                keep_last=keep_last,
                config=checkpoint_config,
                metadata=checkpoint_metadata,
            )
        end
    end

    if ckpt_root !== nothing && ckpt_every > 0 && state.step > 0 && state.step % ckpt_every != 0
        save_trainer_checkpoint(
            state,
            ckpt_root;
            family=family,
            mode=mode,
            keep_last=keep_last,
            config=checkpoint_config,
            metadata=checkpoint_metadata,
        )
    end

    return last_metrics
end
