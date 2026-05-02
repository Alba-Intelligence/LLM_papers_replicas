function bootstrap_training_config(vocab_size::Integer; seq_len::Integer=128, attn_type::String="gqa")
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    return MythosConfig(
        vocab_size=Int(vocab_size),
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=Int(seq_len),
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type=attn_type,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        lora_rank=4,
        act_threshold=0.99f0,
    )
end

function bootstrap_full_model_training_config(vocab_size::Integer; seq_len::Integer=64, attn_type::String="gqa")
    attn_type == "gqa" || throw(ArgumentError("full-model bootstrap currently supports only gqa attention"))
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    return MythosConfig(
        vocab_size=Int(vocab_size),
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=Int(seq_len),
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type=attn_type,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_experts=1,
        n_shared_experts=0,
        n_experts_per_tok=1,
        expert_dim=16,
        lora_rank=4,
        act_threshold=0.99f0,
    )
end

TransformerCore.text_next_token_pairs(texts::AbstractVector{<:AbstractString}, tokenizer::MythosTokenizer, seq_len::Integer) =
    TransformerCore.text_next_token_pairs(texts, text -> encode(tokenizer, text), seq_len)

function _fineweb_runner()
    if haskey(ENV, "OPENMYTHOS_FINEWEB_PYTHON")
        return Cmd([ENV["OPENMYTHOS_FINEWEB_PYTHON"]])
    end
    return Cmd(["uv", "run", "--with", "datasets", "--with", "transformers", "python"])
end

function _run_fineweb_python(model_id::String, seq_len::Integer, batch_size::Integer; subset::String="sample-10BT", max_batches::Integer=8, runner::Cmd=_fineweb_runner())
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    max_batches > 0 || throw(ArgumentError("max_batches must be positive"))

    script = """
import argparse
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

from datasets import load_dataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", required=True)
parser.add_argument("--seq-len", required=True, type=int)
parser.add_argument("--batch-size", required=True, type=int)
parser.add_argument("--subset", default="sample-10BT")
parser.add_argument("--max-batches", required=True, type=int)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_id)
ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True)

buf = []
xs = []
ys = []
produced = 0

for sample in ds:
    buf.extend(tok.encode(sample["text"], add_special_tokens=False))
    while len(buf) >= args.seq_len + 1:
        chunk = buf[: args.seq_len + 1]
        buf = buf[args.seq_len + 1 :]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
        if len(xs) == args.batch_size:
            print(";".join(",".join(str(v) for v in row) for row in xs) + "|" + ";".join(",".join(str(v) for v in row) for row in ys))
            xs = []
            ys = []
            produced += 1
            if produced >= args.max_batches:
                raise SystemExit(0)
"""

    mktemp() do path, io
        write(io, script)
        close(io)
        cmd = `$runner $path --model-id $model_id --seq-len $seq_len --batch-size $batch_size --subset $subset --max-batches $max_batches`
        return read(cmd, String)
    end
end

function _parse_batch_matrix(raw::AbstractString, seq_len::Int)
    isempty(raw) && return Matrix{Int}(undef, 0, seq_len)
    rows = split(raw, ';')
    out = Matrix{Int}(undef, length(rows), seq_len)
    for (i, row) in enumerate(rows)
        vals = parse.(Int, split(row, ','))
        length(vals) == seq_len || error("unexpected sequence length in Python batch response")
        out[i, :] .= vals
    end
    return out
end

function _parse_fineweb_batches(raw::AbstractString, seq_len::Int)
    isempty(strip(raw)) && return Tuple{Matrix{Int}, Matrix{Int}}[]
    batches = Tuple{Matrix{Int}, Matrix{Int}}[]
    for line in split(chomp(raw), '\n')
        isempty(line) && continue
        parts = split(line, '|')
        length(parts) == 2 || error("unexpected Python batch response")
        x = _parse_batch_matrix(parts[1], seq_len)
        y = _parse_batch_matrix(parts[2], seq_len)
        size(x) == size(y) || error("input/target batch shape mismatch")
        push!(batches, (x, y))
    end
    return batches
end

function fineweb_edu_batches(model_id::String, seq_len::Integer, batch_size::Integer; subset::String="sample-10BT", max_batches::Integer=8, runner::Cmd=_fineweb_runner())
    raw = _run_fineweb_python(model_id, seq_len, batch_size; subset=subset, max_batches=max_batches, runner=runner)
    return _parse_fineweb_batches(raw, Int(seq_len))
end

fineweb_edu_batches(tokenizer::MythosTokenizer, seq_len::Integer, batch_size::Integer; kwargs...) =
    fineweb_edu_batches(tokenizer.model_id, seq_len, batch_size; kwargs...)

struct LuxHeadOnlyOpenMythos{M<:OpenMythos} <: Lux.LuxCore.AbstractLuxLayer
    model::M
    n_loops::Union{Nothing, Int}
end

LuxHeadOnlyOpenMythos(model::OpenMythos; n_loops::Union{Nothing, Int}=nothing) = LuxHeadOnlyOpenMythos{typeof(model)}(model, n_loops)

Lux.initialparameters(::AbstractRNG, layer::LuxHeadOnlyOpenMythos) = (head=copy(layer.model.head),)
Lux.initialstates(::AbstractRNG, ::LuxHeadOnlyOpenMythos) = NamedTuple()

function (layer::LuxHeadOnlyOpenMythos)(input_ids::AbstractMatrix{<:Integer}, ps, st)
    hidden = _forward_hidden(layer.model, input_ids; n_loops=layer.n_loops)
    return _linear_feature_last(hidden, ps.head), st
end

mutable struct HeadOnlyTrainerState{T<:AbstractFloat}
    layer::LuxHeadOnlyOpenMythos{OpenMythos{T}}
    head::Matrix{T}
    lux_state::NamedTuple
    opt_state
    schedule::WarmupCosineSchedule{T}
    step::Int
end

function _validate_full_model_cfg(cfg::MythosConfig)
    cfg.attn_type == "gqa" || throw(ArgumentError("full-model training currently supports only gqa attention"))
    cfg.n_experts == 1 || throw(ArgumentError("full-model training currently requires n_experts == 1"))
    cfg.n_shared_experts == 0 || throw(ArgumentError("full-model training currently requires n_shared_experts == 0"))
    cfg.n_experts_per_tok == 1 || throw(ArgumentError("full-model training currently requires n_experts_per_tok == 1"))
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

function _full_model_loss(model::OpenMythos{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    logits = model(input_ids; n_loops=n_loops)
    return _sequence_cross_entropy(logits, target_ids)
end

mutable struct FullModelTrainerState{T<:AbstractFloat}
    model::OpenMythos{T}
    opt_state
    schedule::WarmupCosineSchedule{T}
    step::Int
    n_loops::Union{Nothing, Int}
end

function FullModelTrainerState(
    model::OpenMythos{T};
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
    n_loops::Union{Nothing, Integer}=nothing,
) where {T<:AbstractFloat}
    _validate_full_model_cfg(model.cfg)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(; eta=zero(T), beta=(T(beta1), T(beta2)), lambda=T(weight_decay), epsilon=T(eps)),
        model,
    )
    return FullModelTrainerState(model, opt_state, schedule, 0, n_loops === nothing ? nothing : Int(n_loops))
end

function full_model_logits(state::FullModelTrainerState, input_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing)
    return state.model(input_ids; n_loops=(n_loops === nothing ? state.n_loops : n_loops))
end

function full_model_loss(state::FullModelTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    return _full_model_loss(state.model, input_ids, target_ids; n_loops=(n_loops === nothing ? state.n_loops : n_loops))
end

function train_full_model_step!(state::FullModelTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    size(input_ids) == size(target_ids) || throw(ArgumentError("input_ids and target_ids must have the same shape"))
    active_loops = n_loops === nothing ? state.n_loops : n_loops
    loss, grads = Zygote.withgradient(model -> _full_model_loss(model, input_ids, target_ids; n_loops=active_loops), state.model)
    grad = only(grads)

    lr = learning_rate(state.schedule, state.step)
    Optimisers.adjust!(state.opt_state, lr)
    state.opt_state, state.model = Optimisers.update(state.opt_state, state.model, grad)
    state.step += 1

    return (loss=loss, lr=lr, grad_norm=T(sqrt(_tree_sumsq(grad))), step=state.step)
end

function HeadOnlyTrainerState(
    model::OpenMythos{T};
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
    n_loops::Union{Nothing, Integer}=nothing,
) where {T<:AbstractFloat}
    layer = LuxHeadOnlyOpenMythos(model; n_loops=n_loops)
    head = Lux.initialparameters(Random.default_rng(), layer).head
    lux_state = Lux.initialstates(Random.default_rng(), layer)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(; eta=zero(T), beta=(T(beta1), T(beta2)), lambda=T(weight_decay), epsilon=T(eps)),
        head,
    )
    return HeadOnlyTrainerState(
        layer,
        head,
        lux_state,
        opt_state,
        schedule,
        0,
    )
end

function _layer_for(state::HeadOnlyTrainerState, n_loops::Union{Nothing, Integer})
    if n_loops === nothing || n_loops == state.layer.n_loops
        return state.layer
    end
    return LuxHeadOnlyOpenMythos(state.layer.model; n_loops=n_loops)
end

function head_only_logits(state::HeadOnlyTrainerState, input_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing)
    logits, _ = Lux.apply(_layer_for(state, n_loops), input_ids, (head=state.head,), state.lux_state)
    return logits
end

function head_only_loss(state::HeadOnlyTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    hidden = _forward_hidden(state.layer.model, input_ids; n_loops=(n_loops === nothing ? state.layer.n_loops : n_loops))
    loss, = _head_loss_and_grad(hidden, state.head, target_ids)
    return loss
end

function train_head_only_step!(state::HeadOnlyTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    size(input_ids) == size(target_ids) || throw(ArgumentError("input_ids and target_ids must have the same shape"))
    hidden = _forward_hidden(state.layer.model, input_ids; n_loops=(n_loops === nothing ? state.layer.n_loops : n_loops))
    loss, grad = _head_loss_and_grad(hidden, state.head, target_ids)

    lr = learning_rate(state.schedule, state.step)
    Optimisers.adjust!(state.opt_state, lr)
    state.opt_state, state.head = Optimisers.update(state.opt_state, state.head, grad)
    state.step += 1

    return (loss=loss, lr=lr, grad_norm=T(sqrt(sum(abs2, grad))), step=state.step)
end

function save_head_only_checkpoint(state::HeadOnlyTrainerState, ckpt_dir::AbstractString; keep_last::Integer=3, metadata::AbstractDict=Dict{String, Any}())
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
        "mode" => "lux_head_only",
        "n_loops" => state.layer.n_loops,
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

function save_full_model_checkpoint(state::FullModelTrainerState, ckpt_dir::AbstractString; keep_last::Integer=3, metadata::AbstractDict=Dict{String, Any}())
    keep_last > 0 || throw(ArgumentError("keep_last must be positive"))
    mkpath(ckpt_dir)
    filename = "step_$(lpad(string(state.step), 7, '0')).jls"
    final_path = joinpath(ckpt_dir, filename)
    temp_path = final_path * ".tmp"
    payload = Dict(
        "step" => state.step,
        "model" => state.model,
        "opt_state" => state.opt_state,
        "schedule" => state.schedule,
        "n_loops" => state.n_loops,
        "mode" => "full_model_dense_openmythos",
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

function load_head_only_checkpoint(path::AbstractString, model::OpenMythos{T}) where {T<:AbstractFloat}
    payload = open(path, "r") do io
        deserialize(io)
    end
    head = T.(payload["head"])
    size(head) == size(model.head) || throw(ArgumentError("checkpoint head shape does not match model"))
    n_loops = get(payload, "n_loops", nothing)
    return HeadOnlyTrainerState(
        LuxHeadOnlyOpenMythos(model; n_loops=n_loops),
        head,
        get(payload, "lux_state", NamedTuple()),
        payload["opt_state"],
        payload["schedule"],
        Int(payload["step"]),
    )
end

function load_full_model_checkpoint(path::AbstractString)
    payload = open(path, "r") do io
        deserialize(io)
    end
    get(payload, "mode", nothing) == "full_model_dense_openmythos" || throw(ArgumentError("checkpoint is not an OpenMythos full-model checkpoint"))
    model = payload["model"]
    _validate_full_model_cfg(model.cfg)
    return FullModelTrainerState(
        model,
        payload["opt_state"],
        payload["schedule"],
        Int(payload["step"]),
        get(payload, "n_loops", nothing),
    )
end

function train_head_only!(
    state::HeadOnlyTrainerState,
    batches::AbstractVector{<:Tuple{<:AbstractMatrix{<:Integer}, <:AbstractMatrix{<:Integer}}};
    total_steps::Integer=state.schedule.total_steps,
    n_loops::Union{Nothing, Integer}=nothing,
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
        last_metrics = train_head_only_step!(state, batch[1], batch[2]; n_loops=n_loops)

        if log_every > 0 && (state.step == 1 || state.step % log_every == 0 || state.step == total_steps)
            println(io, "step $(state.step)/$(total_steps) | loss $(round(last_metrics.loss; digits=4)) | gnorm $(round(last_metrics.grad_norm; digits=4)) | lr $(last_metrics.lr)")
        end

        if ckpt_dir !== nothing && ckpt_every > 0 && state.step % ckpt_every == 0
            save_head_only_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
        end
    end

    if ckpt_dir !== nothing && ckpt_every > 0 && state.step > 0 && state.step % ckpt_every != 0
        save_head_only_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
    end

    return last_metrics
end

function train_full_model!(
    state::FullModelTrainerState,
    batches::AbstractVector{<:Tuple{<:AbstractMatrix{<:Integer}, <:AbstractMatrix{<:Integer}}};
    total_steps::Integer=state.schedule.total_steps,
    n_loops::Union{Nothing, Integer}=nothing,
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
        last_metrics = train_full_model_step!(state, batch[1], batch[2]; n_loops=n_loops)

        if log_every > 0 && (state.step == 1 || state.step % log_every == 0 || state.step == total_steps)
            println(io, "step $(state.step)/$(total_steps) | loss $(round(last_metrics.loss; digits=4)) | gnorm $(round(last_metrics.grad_norm; digits=4)) | lr $(last_metrics.lr)")
        end

        if ckpt_dir !== nothing && ckpt_every > 0 && state.step % ckpt_every == 0
            save_full_model_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
        end
    end

    if ckpt_dir !== nothing && ckpt_every > 0 && state.step > 0 && state.step % ckpt_every != 0
        save_full_model_checkpoint(state, ckpt_dir; keep_last=keep_last, metadata=checkpoint_metadata)
    end

    return last_metrics
end
