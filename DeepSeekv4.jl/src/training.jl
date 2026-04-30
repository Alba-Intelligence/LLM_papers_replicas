function bootstrap_deepseek_training_config(vocab_size::Integer; seq_len::Integer=64)
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    return DeepSeekV4Config(
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
    )
end

struct LuxHeadOnlyDeepSeekV4{M<:DeepSeekV4Model} <: Lux.LuxCore.AbstractLuxLayer
    model::M
end

Lux.initialparameters(::AbstractRNG, layer::LuxHeadOnlyDeepSeekV4) = (head=copy(layer.model.head),)
Lux.initialstates(::AbstractRNG, ::LuxHeadOnlyDeepSeekV4) = NamedTuple()

function (layer::LuxHeadOnlyDeepSeekV4)(input_ids::AbstractMatrix{<:Integer}, ps, st)
    hidden = deepseek_hidden(layer.model, input_ids)
    return _linear_feature_last(hidden, ps.head), st
end

mutable struct DeepSeekHeadTrainerState{T<:AbstractFloat}
    layer::LuxHeadOnlyDeepSeekV4{DeepSeekV4Model{T}}
    head::Matrix{T}
    lux_state::NamedTuple
    opt_state
    schedule::WarmupCosineSchedule{T}
    step::Int
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
    head = Lux.initialparameters(Random.default_rng(), layer).head
    lux_state = Lux.initialstates(Random.default_rng(), layer)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(; eta=zero(T), beta=(T(beta1), T(beta2)), lambda=T(weight_decay), epsilon=T(eps)),
        head,
    )
    return DeepSeekHeadTrainerState(
        layer,
        head,
        lux_state,
        opt_state,
        schedule,
        0,
    )
end

function deepseek_head_logits(state::DeepSeekHeadTrainerState, input_ids::AbstractMatrix{<:Integer})
    logits, _ = Lux.apply(state.layer, input_ids, (head=state.head,), state.lux_state)
    return logits
end

function deepseek_head_loss(state::DeepSeekHeadTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    hidden = deepseek_hidden(state.layer.model, input_ids)
    loss, = _head_loss_and_grad(hidden, state.head, target_ids)
    return loss
end

function train_deepseek_head_only_step!(state::DeepSeekHeadTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    size(input_ids) == size(target_ids) || throw(ArgumentError("input_ids and target_ids must have the same shape"))
    hidden = deepseek_hidden(state.layer.model, input_ids)
    loss, grad = _head_loss_and_grad(hidden, state.head, target_ids)

    lr = learning_rate(state.schedule, state.step)
    Optimisers.adjust!(state.opt_state, lr)
    state.opt_state, state.head = Optimisers.update(state.opt_state, state.head, grad)
    state.step += 1

    return (loss=loss, lr=lr, grad_norm=T(sqrt(sum(abs2, grad))), step=state.step)
end

function save_deepseek_checkpoint(state::DeepSeekHeadTrainerState, ckpt_dir::AbstractString; keep_last::Integer=3, metadata::AbstractDict=Dict{String, Any}())
    keep_last > 0 || throw(ArgumentError("keep_last must be positive"))
    mkpath(ckpt_dir)
    filename = "step_$(lpad(string(state.step), 7, '0')).jls"
    final_path = joinpath(ckpt_dir, filename)
    temp_path = final_path * ".tmp"
    payload = Dict(
        "step" => state.step,
        "head" => copy(state.head),
        "lux_state" => state.lux_state,
        "opt_state" => state.opt_state,
        "schedule" => state.schedule,
        "cfg" => state.layer.model.cfg,
        "vocab_size" => state.layer.model.cfg.vocab_size,
        "mode" => "lux_head_only_deepseek",
        "metadata" => Dict(string(k) => v for (k, v) in pairs(metadata)),
    )
    open(temp_path, "w") do io
        serialize(io, payload)
    end
    mv(temp_path, final_path; force=true)

    existing = filter(name -> startswith(name, "step_") && endswith(name, ".jls"), readdir(ckpt_dir))
    sort!(existing)
    for old in existing[1:max(0, length(existing) - keep_last)]
        rm(joinpath(ckpt_dir, old); force=true)
    end

    return final_path
end

function load_deepseek_checkpoint(path::AbstractString, model::DeepSeekV4Model{T}) where {T<:AbstractFloat}
    payload = open(path, "r") do io
        deserialize(io)
    end
    head = T.(payload["head"])
    size(head) == size(model.head) || throw(ArgumentError("checkpoint head shape does not match model"))
    return DeepSeekHeadTrainerState(
        LuxHeadOnlyDeepSeekV4(model),
        head,
        get(payload, "lux_state", NamedTuple()),
        payload["opt_state"],
        payload["schedule"],
        Int(payload["step"]),
    )
end

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
    isempty(batches) && throw(ArgumentError("at least one batch is required"))
    total_steps >= state.step || throw(ArgumentError("total_steps must be >= current step"))

    last_metrics = (loss=zero(eltype(state.head)), lr=zero(eltype(state.head)), grad_norm=zero(eltype(state.head)), step=state.step)
    while state.step < total_steps
        batch = batches[mod1(state.step + 1, length(batches))]
        last_metrics = train_deepseek_head_only_step!(state, batch[1], batch[2])

        if log_every > 0 && (state.step == 1 || state.step % log_every == 0 || state.step == total_steps)
            println(io, "step $(state.step)/$(total_steps) | loss $(round(last_metrics.loss; digits=4)) | gnorm $(round(last_metrics.grad_norm; digits=4)) | lr $(last_metrics.lr)")
        end

        if ckpt_dir !== nothing && ckpt_every > 0 && state.step % ckpt_every == 0
            save_deepseek_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
        end
    end

    if ckpt_dir !== nothing && ckpt_every > 0 && state.step > 0 && state.step % ckpt_every != 0
        save_deepseek_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
    end

    return last_metrics
end
