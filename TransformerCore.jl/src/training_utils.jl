"""
    WarmupCosineSchedule{T}

Learning-rate schedule with linear warmup followed by cosine decay.
"""
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

"""
    learning_rate(schedule, step)

Return the learning rate for `step` under `schedule`.
"""
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

"""
    chunk_next_token_pairs(token_ids, seq_len)

Split a flat token stream into `(input, target)` next-token training pairs of
length `seq_len`.
"""
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

"""
    text_next_token_pairs(texts, encode_fn, seq_len)

Encode `texts` incrementally with `encode_fn` and drain them into next-token
 `(input, target)` training pairs of length `seq_len`.
"""
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

"""
    batch_next_token_pairs(pairs, batch_size; drop_last=false)

Pack token-pair sequences into batched `(x, y)` matrices.
"""
function batch_next_token_pairs(
    pairs::AbstractVector{<:Tuple{<:AbstractVector{<:Integer}, <:AbstractVector{<:Integer}}},
    batch_size::Integer;
    drop_last::Bool=false,
)
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

"""
    _head_loss_and_grad(hidden, head, target_ids)

Compute cross-entropy loss and the LM-head weight gradient for a fixed hidden
state tensor.

This helper is used by the head-only bootstrap training paths in both model
families.
"""
function _head_loss_and_grad(hidden::AbstractArray{T}, head::AbstractMatrix{T}, target_ids::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    flat_hidden = _flatten_feature_last(hidden)
    logits = head * flat_hidden
    logits_rows = permutedims(logits, (2, 1))
    log_probs = similar(logits_rows)

    for i in axes(logits_rows, 1)
        row = @view logits_rows[i, :]
        m = maximum(row)
        shifted = row .- m
        log_denom = log(sum(exp.(shifted)))
        @view(log_probs[i, :]) .= shifted .- log_denom
    end

    probs = exp.(log_probs)
    flat_targets = vec(target_ids)
    total_loss = zero(T)

    for i in axes(logits_rows, 1)
        tgt = flat_targets[i] + 1
        (1 <= tgt <= size(probs, 2)) || throw(BoundsError(probs, (i, tgt)))
        total_loss -= log_probs[i, tgt]
    end

    n = size(logits_rows, 1)
    grad_rows = probs
    for i in eachindex(flat_targets)
        grad_rows[i, flat_targets[i] + 1] -= one(T)
    end
    grad_rows ./= T(n)

    grad_weight = permutedims(grad_rows, (2, 1)) * permutedims(flat_hidden, (2, 1))
    return total_loss / T(n), grad_weight
end
