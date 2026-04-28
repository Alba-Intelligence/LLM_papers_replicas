struct WarmupCosineSchedule{T<:AbstractFloat}
    warmup_steps::Int
    total_steps::Int
    max_lr::T
    min_lr::T
end

function WarmupCosineSchedule(
    warmup_steps::Integer,
    total_steps::Integer,
    max_lr::Real,
    min_lr::Real=max_lr * 0.1,
)
    warmup_steps >= 0 || throw(ArgumentError("warmup_steps must be non-negative"))
    total_steps > 0 || throw(ArgumentError("total_steps must be positive"))
    total_steps >= warmup_steps || throw(ArgumentError("total_steps must be >= warmup_steps"))
    max_lr > 0 || throw(ArgumentError("max_lr must be positive"))
    min_lr >= 0 || throw(ArgumentError("min_lr must be non-negative"))
    T = promote_type(typeof(float(max_lr)), typeof(float(min_lr)))
    return WarmupCosineSchedule{T}(Int(warmup_steps), Int(total_steps), T(max_lr), T(min_lr))
end

function learning_rate(schedule::WarmupCosineSchedule{T}, step::Integer) where {T}
    step < 0 && throw(ArgumentError("step must be non-negative"))
    if schedule.warmup_steps > 0 && step < schedule.warmup_steps
        return schedule.max_lr * (T(step) / T(schedule.warmup_steps))
    end
    if step >= schedule.total_steps
        return schedule.min_lr
    end
    if schedule.total_steps == schedule.warmup_steps
        return schedule.min_lr
    end
    decay = T(step - schedule.warmup_steps) / T(schedule.total_steps - schedule.warmup_steps)
    return schedule.min_lr + T(0.5) * (schedule.max_lr - schedule.min_lr) * (one(T) + cos(T(pi) * decay))
end

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

function chunk_next_token_pairs(token_ids::AbstractVector{<:Integer}, seq_len::Integer)
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    pairs = Tuple{Vector{Int}, Vector{Int}}[]
    idx = firstindex(token_ids)
    last_needed = lastindex(token_ids) - seq_len
    while idx <= last_needed
        chunk = Int.(token_ids[idx:(idx + seq_len)])
        push!(pairs, (copy(chunk[1:end-1]), copy(chunk[2:end])))
        idx += seq_len + 1
    end
    return pairs
end

function _drain_token_buffer!(pairs::Vector{Tuple{Vector{Int}, Vector{Int}}}, buffer::Vector{Int}, seq_len::Int)
    while length(buffer) >= seq_len + 1
        chunk = buffer[1:(seq_len + 1)]
        push!(pairs, (copy(chunk[1:end-1]), copy(chunk[2:end])))
        deleteat!(buffer, 1:(seq_len + 1))
    end
    return pairs
end

function text_next_token_pairs(texts::AbstractVector{<:AbstractString}, encode_fn::Function, seq_len::Integer)
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    pairs = Tuple{Vector{Int}, Vector{Int}}[]
    buffer = Int[]
    for text in texts
        append!(buffer, Int.(encode_fn(text)))
        _drain_token_buffer!(pairs, buffer, Int(seq_len))
    end
    return pairs
end

text_next_token_pairs(texts::AbstractVector{<:AbstractString}, tokenizer::MythosTokenizer, seq_len::Integer) =
    text_next_token_pairs(texts, text -> encode(tokenizer, text), seq_len)

function batch_next_token_pairs(pairs::AbstractVector{<:Tuple{<:AbstractVector{<:Integer}, <:AbstractVector{<:Integer}}}, batch_size::Integer; drop_last::Bool=false)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    isempty(pairs) && return Tuple{Matrix{Int}, Matrix{Int}}[]

    seq_len = length(first(pairs)[1])
    batches = Tuple{Matrix{Int}, Matrix{Int}}[]
    start = firstindex(pairs)

    while start <= lastindex(pairs)
        stop = min(start + batch_size - 1, lastindex(pairs))
        current = @view pairs[start:stop]
        if drop_last && length(current) < batch_size
            break
        end
        x = Matrix{Int}(undef, length(current), seq_len)
        y = Matrix{Int}(undef, length(current), seq_len)
        for (i, (input_ids, target_ids)) in enumerate(current)
            length(input_ids) == seq_len || throw(ArgumentError("inconsistent input sequence length"))
            length(target_ids) == seq_len || throw(ArgumentError("inconsistent target sequence length"))
            x[i, :] .= input_ids
            y[i, :] .= target_ids
        end
        push!(batches, (x, y))
        start = stop + 1
    end

    return batches
end

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

mutable struct HeadOnlyTrainerState{T<:AbstractFloat}
    model::OpenMythos{T}
    head::Matrix{T}
    adam_m::Matrix{T}
    adam_v::Matrix{T}
    schedule::WarmupCosineSchedule{T}
    weight_decay::T
    beta1::T
    beta2::T
    eps::T
    step::Int
end

function HeadOnlyTrainerState(
    model::OpenMythos{T};
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
) where {T<:AbstractFloat}
    head = copy(model.head)
    zeros_like = zero.(head)
    return HeadOnlyTrainerState(
        model,
        head,
        copy(zeros_like),
        copy(zeros_like),
        schedule,
        T(weight_decay),
        T(beta1),
        T(beta2),
        T(eps),
        0,
    )
end

function _head_loss_and_grad(hidden::AbstractArray{T}, head::AbstractMatrix{T}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    flat_hidden = _flatten_feature_last(hidden)
    logits = head * flat_hidden
    logits_rows = permutedims(logits, (2, 1))
    probs = similar(logits_rows)
    flat_targets = vec(target_ids)
    total_loss = zero(T)

    for i in axes(logits_rows, 1)
        row = @view logits_rows[i, :]
        m = maximum(row)
        exps = exp.(row .- m)
        s = sum(exps)
        @view(probs[i, :]) .= exps ./ s
        tgt = flat_targets[i] + 1
        (1 <= tgt <= size(probs, 2)) || throw(BoundsError(probs, (i, tgt)))
        total_loss -= log(probs[i, tgt])
    end

    n = size(logits_rows, 1)
    grad_rows = copy(probs)
    for i in eachindex(flat_targets)
        grad_rows[i, flat_targets[i] + 1] -= one(T)
    end
    grad_rows ./= T(n)

    grad_weight = permutedims(grad_rows, (2, 1)) * permutedims(flat_hidden, (2, 1))
    return total_loss / T(n), grad_weight
end

function head_only_logits(state::HeadOnlyTrainerState, input_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing)
    hidden = _forward_hidden(state.model, input_ids; n_loops=n_loops)
    return _linear_feature_last(hidden, state.head)
end

function head_only_loss(state::HeadOnlyTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    hidden = _forward_hidden(state.model, input_ids; n_loops=n_loops)
    loss, = _head_loss_and_grad(hidden, state.head, target_ids)
    return loss
end

function train_head_only_step!(state::HeadOnlyTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    size(input_ids) == size(target_ids) || throw(ArgumentError("input_ids and target_ids must have the same shape"))
    hidden = _forward_hidden(state.model, input_ids; n_loops=n_loops)
    loss, grad = _head_loss_and_grad(hidden, state.head, target_ids)

    lr = learning_rate(state.schedule, state.step)
    state.step += 1

    one_minus_beta1 = one(T) - state.beta1
    one_minus_beta2 = one(T) - state.beta2
    state.adam_m .= state.beta1 .* state.adam_m .+ one_minus_beta1 .* grad
    state.adam_v .= state.beta2 .* state.adam_v .+ one_minus_beta2 .* (grad .* grad)

    bias1 = one(T) - state.beta1 ^ state.step
    bias2 = one(T) - state.beta2 ^ state.step
    mhat = state.adam_m ./ bias1
    vhat = state.adam_v ./ bias2

    if state.weight_decay != 0
        state.head .*= one(T) - lr * state.weight_decay
    end
    state.head .-= lr .* (mhat ./ (sqrt.(vhat) .+ state.eps))

    return (loss=loss, lr=lr, grad_norm=T(sqrt(sum(abs2, grad))), step=state.step)
end

function latest_checkpoint(ckpt_dir::AbstractString)
    isdir(ckpt_dir) || return nothing
    matches = filter(name -> startswith(name, "step_") && endswith(name, ".jls"), readdir(ckpt_dir))
    isempty(matches) && return nothing
    sort!(matches)
    return joinpath(ckpt_dir, last(matches))
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
        "adam_m" => copy(state.adam_m),
        "adam_v" => copy(state.adam_v),
        "schedule" => state.schedule,
        "weight_decay" => state.weight_decay,
        "beta1" => state.beta1,
        "beta2" => state.beta2,
        "eps" => state.eps,
        "cfg" => state.model.cfg,
        "vocab_size" => state.model.cfg.vocab_size,
        "mode" => "head_only",
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
    return HeadOnlyTrainerState(
        model,
        head,
        T.(payload["adam_m"]),
        T.(payload["adam_v"]),
        payload["schedule"],
        T(payload["weight_decay"]),
        T(payload["beta1"]),
        T(payload["beta2"]),
        T(payload["eps"]),
        Int(payload["step"]),
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
