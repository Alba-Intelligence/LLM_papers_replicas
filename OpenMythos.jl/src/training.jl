"""
    bootstrap_training_config(vocab_size; seq_len=128, attn_type="gqa")

Return a small OpenMythos configuration for head-only bootstrap training and
smoke tests.
"""
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

"""
    bootstrap_full_model_training_config(vocab_size; seq_len=64, attn_type="gqa", n_experts=1, n_shared_experts=0, n_experts_per_tok=1)

Return a constrained dense OpenMythos configuration suitable for the current
full-model bootstrap trainer.

The trainer now supports both GQA and MLA variants across small dense or sparse
OpenMythos bootstrap configs. Shared experts remain optional.
"""
function bootstrap_full_model_training_config(
    vocab_size::Integer;
    seq_len::Integer=64,
    attn_type::String="gqa",
    n_experts::Integer=1,
    n_shared_experts::Integer=0,
    n_experts_per_tok::Integer=1,
)
    attn_type in ("gqa", "mla") || throw(ArgumentError("full-model bootstrap currently supports only gqa or mla attention"))
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    n_experts > 0 || throw(ArgumentError("n_experts must be positive"))
    n_shared_experts >= 0 || throw(ArgumentError("n_shared_experts must be non-negative"))
    n_experts_per_tok > 0 || throw(ArgumentError("n_experts_per_tok must be positive"))
    n_experts_per_tok <= n_experts || throw(ArgumentError("n_experts_per_tok must be <= n_experts"))
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
        n_experts=Int(n_experts),
        n_shared_experts=Int(n_shared_experts),
        n_experts_per_tok=Int(n_experts_per_tok),
        expert_dim=16,
        lora_rank=4,
        act_threshold=0.99f0,
    )
end

TransformerCore.text_next_token_pairs(texts::AbstractVector{<:AbstractString}, tokenizer::MythosTokenizer, seq_len::Integer) =
    TransformerCore.text_next_token_pairs(texts, text -> encode(tokenizer, text), seq_len)

const _FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
const _FINEWEB_ROWS_API = "https://datasets-server.huggingface.co/rows"
const _FINEWEB_MAX_ROWS_PER_REQUEST = 100

function _download_json(url::AbstractString)
    mktemp() do path, io
        close(io)
        Downloads.download(url, path)
        return JSON3.read(read(path, String))
    end
end

function _drain_next_token_pairs!(pairs, buffer::Vector{Int}, seq_len::Int, max_pairs::Int)
    while length(buffer) >= seq_len + 1 && length(pairs) < max_pairs
        chunk = buffer[1:(seq_len + 1)]
        push!(pairs, (copy(chunk[1:end-1]), copy(chunk[2:end])))
        deleteat!(buffer, 1:(seq_len + 1))
    end
    return pairs
end

function _fineweb_rows_texts(; dataset::AbstractString=_FINEWEB_DATASET, config::AbstractString="sample-10BT", split::AbstractString="train", offset::Integer=0, length::Integer=_FINEWEB_MAX_ROWS_PER_REQUEST, fetch_json::Function=_download_json)
    0 <= offset || throw(ArgumentError("offset must be non-negative"))
    1 <= length <= _FINEWEB_MAX_ROWS_PER_REQUEST || throw(ArgumentError("length must be between 1 and $(_FINEWEB_MAX_ROWS_PER_REQUEST)"))

    url = "$(_FINEWEB_ROWS_API)?dataset=$(dataset)&config=$(config)&split=$(split)&offset=$(Int(offset))&length=$(Int(length))"
    payload = fetch_json(url)
    texts = String[]
    for entry in payload.rows
        row = entry.row
        hasproperty(row, :text) || continue
        push!(texts, String(row.text))
    end
    return texts
end

"""
    fineweb_edu_batches_from_parquet(tokenizer, parquet_path, seq_len, batch_size; max_batches=8)

Build small next-token training batches from local FineWeb-Edu parquet shard(s)
using the shared Julia-native text-data helpers.
"""
function fineweb_edu_batches_from_parquet(
    tokenizer::MythosTokenizer,
    parquet_path::AbstractString,
    seq_len::Integer,
    batch_size::Integer;
    max_batches::Integer=8,
)
    return TextDataCore.next_token_batches_from_parquet(
        _native_tokenizer(tokenizer),
        parquet_path,
        seq_len,
        batch_size;
        max_batches=max_batches,
    )
end

fineweb_edu_batches_from_parquet(model_id::String, parquet_path::AbstractString, seq_len::Integer, batch_size::Integer; kwargs...) =
    fineweb_edu_batches_from_parquet(MythosTokenizer(model_id), parquet_path, seq_len, batch_size; kwargs...)

"""
    fineweb_edu_batches(tokenizer, seq_len, batch_size; subset="sample-10BT", max_batches=8)

Fetch small next-token training batches from FineWeb-Edu through the Julia-native
Hugging Face dataset viewer rows API. This is intended for smoke-scale remote
training; larger local parquet shards are still the preferred Julia path.
"""
function fineweb_edu_batches(
    tokenizer::MythosTokenizer,
    seq_len::Integer,
    batch_size::Integer;
    subset::String="sample-10BT",
    split::String="train",
    max_batches::Integer=8,
    fetch_rows::Function=_fineweb_rows_texts,
)
    seq_len > 0 || throw(ArgumentError("seq_len must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    max_batches > 0 || throw(ArgumentError("max_batches must be positive"))

    max_pairs = Int(max_batches) * Int(batch_size)
    pairs = Tuple{Vector{Int}, Vector{Int}}[]
    buffer = Int[]
    offset = 0

    while length(pairs) < max_pairs
        texts = fetch_rows(; dataset=_FINEWEB_DATASET, config=subset, split=split, offset=offset, length=_FINEWEB_MAX_ROWS_PER_REQUEST)
        isempty(texts) && break
        for text in texts
            append!(buffer, encode(tokenizer, text))
            _drain_next_token_pairs!(pairs, buffer, Int(seq_len), max_pairs)
            length(pairs) >= max_pairs && break
        end
        offset += length(texts)
        length(texts) < _FINEWEB_MAX_ROWS_PER_REQUEST && break
    end

    return batch_next_token_pairs(pairs, batch_size; drop_last=false)
end

fineweb_edu_batches(model_id::String, seq_len::Integer, batch_size::Integer; kwargs...) =
    fineweb_edu_batches(MythosTokenizer(model_id), seq_len, batch_size; kwargs...)

"""
    LuxHeadOnlyOpenMythos

Lux wrapper that exposes only the OpenMythos LM head as trainable parameters.
"""
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

"""
    HeadOnlyTrainerState{T}

Mutable training state for the Lux-backed OpenMythos head-only bootstrap path.
"""
mutable struct HeadOnlyTrainerState{T<:AbstractFloat}
    layer::LuxHeadOnlyOpenMythos{OpenMythos{T}}
    head::Matrix{T}
    lux_state::NamedTuple
    opt_state
    schedule::WarmupCosineSchedule{T}
    step::Int
end

function _validate_full_model_cfg(cfg::MythosConfig)
    cfg.attn_type in ("gqa", "mla") || throw(ArgumentError("full-model training currently supports only gqa or mla attention"))
    cfg.n_experts > 0 || throw(ArgumentError("full-model training requires n_experts > 0"))
    cfg.n_shared_experts >= 0 || throw(ArgumentError("full-model training requires n_shared_experts >= 0"))
    cfg.n_experts_per_tok > 0 || throw(ArgumentError("full-model training requires n_experts_per_tok > 0"))
    cfg.n_experts_per_tok <= cfg.n_experts || throw(ArgumentError("full-model training requires n_experts_per_tok <= n_experts"))
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

"""
    FullModelTrainerState{T}

Mutable training state for the current dense OpenMythos full-model bootstrap
trainer.
"""
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

"""Return LM logits from the full-model trainer state."""
function full_model_logits(state::FullModelTrainerState, input_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing)
    return state.model(input_ids; n_loops=(n_loops === nothing ? state.n_loops : n_loops))
end

"""Return sequence cross-entropy for the full-model trainer state."""
function full_model_loss(state::FullModelTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    return _full_model_loss(state.model, input_ids, target_ids; n_loops=(n_loops === nothing ? state.n_loops : n_loops))
end

"""Take one optimization step in the full-model OpenMythos trainer."""
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

struct LuxConfiguredOpenMythos{L<:LuxOpenMythos} <: Lux.LuxCore.AbstractLuxLayer
    model::L
    n_loops::Union{Nothing, Int}
end

LuxConfiguredOpenMythos(model::LuxOpenMythos; n_loops::Union{Nothing, Integer}=nothing) =
    LuxConfiguredOpenMythos{typeof(model)}(model, n_loops === nothing ? nothing : Int(n_loops))

Lux.initialparameters(rng::AbstractRNG, layer::LuxConfiguredOpenMythos) = Lux.initialparameters(rng, layer.model)
Lux.initialstates(rng::AbstractRNG, layer::LuxConfiguredOpenMythos) = Lux.initialstates(rng, layer.model)

function (layer::LuxConfiguredOpenMythos)(input_ids::AbstractMatrix{<:Integer}, ps, st)
    return Lux.apply(layer.model, (input_ids=input_ids, n_loops=layer.n_loops), ps, st)
end

"""
    LuxFullModelTrainerState

Lux-native OpenMythos full-model trainer state built on the shared
`TransformerCore.NextTokenTrainerState` foundation.
"""
mutable struct LuxFullModelTrainerState{L,S}
    layer::L
    trainer::S
end

function LuxFullModelTrainerState(
    model::OpenMythos{T};
    schedule::WarmupCosineSchedule{T}=WarmupCosineSchedule(0, 1, T(1e-3), zero(T)),
    weight_decay::Real=0.1,
    beta1::Real=0.9,
    beta2::Real=0.95,
    eps::Real=1e-8,
    n_loops::Union{Nothing, Integer}=nothing,
) where {T<:AbstractFloat}
    _validate_full_model_cfg(model.cfg)
    active_loops = n_loops === nothing ? model.cfg.max_loop_iters : Int(n_loops)
    layer = LuxConfiguredOpenMythos(LuxOpenMythos(model.cfg); n_loops=active_loops)
    ps = to_lux_parameters(model)
    st = Lux.initialstates(Random.default_rng(), layer)
    opt_state = Optimisers.setup(
        Optimisers.AdamW(; eta=zero(T), beta=(T(beta1), T(beta2)), lambda=T(weight_decay), epsilon=T(eps)),
        ps,
    )
    trainer = TransformerCore.NextTokenTrainerState(ps, st, opt_state, schedule, 0)
    return LuxFullModelTrainerState(layer, trainer)
end

"""Return LM logits from the Lux-native full-model trainer state."""
function lux_full_model_logits(state::LuxFullModelTrainerState, input_ids::AbstractMatrix{<:Integer})
    return TransformerCore.next_token_logits(state.layer, state.trainer, input_ids)
end

"""Return sequence cross-entropy from the Lux-native full-model trainer state."""
function lux_full_model_loss(state::LuxFullModelTrainerState, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer})
    return TransformerCore.next_token_loss(state.layer, state.trainer, input_ids, target_ids)
end

"""Take one optimization step in the Lux-native OpenMythos full-model trainer."""
function train_lux_full_model_step!(state::LuxFullModelTrainerState, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer})
    return TransformerCore.train_next_token_step!(state.trainer, state.layer, input_ids, target_ids)
end

"""
    train_lux_full_model!(state, batches; ...)

Run the Lux-native OpenMythos full-model trainer over token batches. When
`ckpt_root` is provided, shared family/mode-aware checkpoints are written under
`openmythos/full_model_lux/`.
"""
function train_lux_full_model!(
    state::LuxFullModelTrainerState,
    batches::AbstractVector{<:Tuple{<:AbstractMatrix{<:Integer}, <:AbstractMatrix{<:Integer}}};
    total_steps::Integer=state.trainer.schedule.total_steps,
    log_every::Integer=10,
    ckpt_root::Union{Nothing, AbstractString}=nothing,
    ckpt_every::Integer=0,
    keep_last::Integer=3,
    io::IO=stdout,
    checkpoint_metadata::AbstractDict=Dict{String, Any}(),
)
    metadata = Dict{String, Any}(string(k) => v for (k, v) in pairs(checkpoint_metadata))
    metadata["n_loops"] = something(state.layer.n_loops, state.layer.model.cfg.max_loop_iters)
    return TransformerCore.train_next_token!(
        state.trainer,
        state.layer,
        batches;
        total_steps=total_steps,
        log_every=log_every,
        ckpt_root=ckpt_root,
        family="openmythos",
        mode="full_model_lux",
        ckpt_every=ckpt_every,
        keep_last=keep_last,
        io=io,
        checkpoint_config=state.layer.model.cfg,
        checkpoint_metadata=metadata,
    )
end

"""Save a Lux-native full-model checkpoint under the shared family/mode layout."""
function save_lux_full_model_checkpoint(
    state::LuxFullModelTrainerState,
    ckpt_root::AbstractString;
    keep_last::Integer=3,
    metadata::AbstractDict=Dict{String, Any}(),
)
    merged = Dict{String, Any}(string(k) => v for (k, v) in pairs(metadata))
    merged["n_loops"] = something(state.layer.n_loops, state.layer.model.cfg.max_loop_iters)
    return TransformerCore.save_trainer_checkpoint(
        state.trainer,
        ckpt_root;
        family="openmythos",
        mode="full_model_lux",
        keep_last=keep_last,
        config=state.layer.model.cfg,
        metadata=merged,
    )
end

"""Restore a Lux-native full-model trainer state from a shared checkpoint."""
function load_lux_full_model_checkpoint(path::AbstractString; n_loops::Union{Nothing, Integer}=nothing)
    restored = TransformerCore.load_trainer_checkpoint(path; expected_family="openmythos", expected_mode="full_model_lux")
    cfg = restored.config isa MythosConfig ? restored.config : throw(ArgumentError("checkpoint config is not a MythosConfig"))
    metadata = restored.metadata isa AbstractDict ? restored.metadata : Dict{String, Any}()
    active_loops = n_loops === nothing ? get(metadata, "n_loops", cfg.max_loop_iters) : Int(n_loops)
    layer = LuxConfiguredOpenMythos(LuxOpenMythos(cfg); n_loops=Int(active_loops))
    return LuxFullModelTrainerState(layer, restored.state)
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

"""Return LM logits from the head-only trainer state."""
function head_only_logits(state::HeadOnlyTrainerState, input_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing)
    logits, _ = Lux.apply(_layer_for(state, n_loops), input_ids, (head=state.head,), state.lux_state)
    return logits
end

"""Return sequence cross-entropy for the head-only trainer state."""
function head_only_loss(state::HeadOnlyTrainerState{T}, input_ids::AbstractMatrix{<:Integer}, target_ids::AbstractMatrix{<:Integer}; n_loops::Union{Nothing, Integer}=nothing) where {T<:AbstractFloat}
    hidden = _forward_hidden(state.layer.model, input_ids; n_loops=(n_loops === nothing ? state.layer.n_loops : n_loops))
    loss, = _head_loss_and_grad(hidden, state.head, target_ids)
    return loss
end

"""Take one optimization step in the OpenMythos head-only trainer."""
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

"""Serialize the head-only trainer state to a checkpoint directory."""
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

"""Serialize the full-model trainer state to a checkpoint directory."""
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

"""Restore a head-only trainer state from `path` for `model`."""
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

"""Restore a dense full-model trainer state from `path`."""
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

"""Run a multi-step head-only OpenMythos training loop over `batches`."""
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

"""Run a multi-step dense full-model OpenMythos training loop over `batches`."""
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
