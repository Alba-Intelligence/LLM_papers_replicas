function loop_index_embedding(h::AbstractArray, loop_t::Integer, loop_dim::Integer; theta::Real=10_000.0f0)
    loop_dim > 0 || throw(ArgumentError("loop_dim must be positive"))
    iseven(loop_dim) || throw(ArgumentError("loop_dim must be even"))
    loop_dim <= size(h, ndims(h)) || throw(DimensionMismatch("loop_dim exceeds hidden dimension"))

    work_t = promote_type(Float32, eltype(h))
    idx = work_t.(0:2:(loop_dim - 2))
    freqs = one(work_t) ./ (work_t(theta) .^ (idx ./ work_t(loop_dim)))
    angles = work_t(loop_t) .* freqs
    emb = vcat(sin.(angles), cos.(angles))
    emb_full = zeros(work_t, size(h, ndims(h)))
    emb_full[1:loop_dim] .= emb
    return h .+ _feature_broadcast(emb_full, ndims(h))
end

struct LoRAAdapter{T<:AbstractFloat}
    down::Matrix{T}
    B::Matrix{T}
    scale::Matrix{T}
end

function LoRAAdapter(dim::Integer, rank::Integer, max_loops::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    rank > 0 || throw(ArgumentError("rank must be positive"))
    max_loops > 0 || throw(ArgumentError("max_loops must be positive"))

    return LoRAAdapter(
        T.(0.02 .* randn(rng, rank, dim)),
        T.(0.02 .* randn(rng, dim, rank)),
        T.(0.02 .* randn(rng, rank, max_loops)),
    )
end

function (adapter::LoRAAdapter)(x::AbstractArray, loop_t::Integer)
    t_idx = clamp(loop_t + 1, 1, size(adapter.scale, 2))
    flat = _flatten_feature_last(x)
    down = adapter.down * flat
    scaled = down .* reshape(adapter.scale[:, t_idx], :, 1)
    out = adapter.B * scaled
    return _unflatten_feature_last(out, size(x))
end

struct LTIInjection{T<:AbstractFloat}
    log_A::Vector{T}
    log_dt::Vector{T}
    B::Vector{T}
end

function LTIInjection(dim::Integer; T::Type{<:AbstractFloat}=Float32)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    return LTIInjection(zeros(T, dim), zeros(T, 1), fill(T(0.1), dim))
end

function get_A(inj::LTIInjection)
    raw = exp.(-exp.(clamp.(inj.log_dt .+ inj.log_A, -20, 20)))
    return max.(raw, floatmin(eltype(raw)))
end

function (inj::LTIInjection)(h::AbstractArray, e::AbstractArray, transformer_out::AbstractArray)
    size(h) == size(e) == size(transformer_out) || throw(DimensionMismatch("all inputs must share shape"))
    A = _feature_broadcast(get_A(inj), ndims(h))
    B = _feature_broadcast(inj.B, ndims(h))
    return A .* h .+ B .* e .+ transformer_out
end

struct ACTHalting{T<:AbstractFloat}
    weight::Matrix{T}
    bias::Vector{T}
end

function ACTHalting(dim::Integer; rng::AbstractRNG=Random.default_rng(), T::Type{<:AbstractFloat}=Float32)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    return ACTHalting(T.(0.02 .* randn(rng, 1, dim)), zeros(T, 1))
end

function (act::ACTHalting)(h::AbstractArray)
    logits = _linear_feature_last(h, act.weight, act.bias)
    return dropdims(_sigmoid.(logits); dims=ndims(logits))
end

struct RecurrentBlock
    cfg::MythosConfig
    block::TransformerBlock
    injection::LTIInjection{Float32}
    act::ACTHalting{Float32}
    lora::LoRAAdapter{Float32}
    norm::RMSNorm{Float32}
    loop_dim::Int
end

function RecurrentBlock(cfg::MythosConfig; rng::AbstractRNG=Random.default_rng())
    return RecurrentBlock(
        cfg,
        TransformerBlock(cfg; use_moe=true, rng=rng),
        LTIInjection(cfg.dim),
        ACTHalting(cfg.dim; rng=rng),
        LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters; rng=rng),
        RMSNorm(cfg.dim),
        cfg.dim ÷ 8,
    )
end

function (recurrent::RecurrentBlock)(h::AbstractArray{T, 3}, e::AbstractArray{T, 3}, freqs_cis::AbstractMatrix; mask=nothing, n_loops::Union{Nothing, Integer}=nothing, kv_cache::Union{Nothing, AbstractDict}=nothing) where {T}
    loops = something(n_loops, recurrent.cfg.max_loop_iters)
    b, t, d = size(h)
    halted = falses(b, t)
    cumulative_p = zeros(Float32, b, t)
    h_out = zeros(T, b, t, d)

    for loop_t in 0:(loops - 1)
        h_loop = loop_index_embedding(h, loop_t, recurrent.loop_dim)
        combined = recurrent.norm(h_loop .+ e)
        cache_key = "recurrent_loop_$(loop_t)"
        trans_out = recurrent.block(combined, freqs_cis; mask=mask, kv_cache=kv_cache, cache_key=cache_key)
        trans_out = trans_out .+ recurrent.lora(trans_out, loop_t)
        h = recurrent.injection(h, e, trans_out)

        p = recurrent.act(h)
        still_running = .!halted
        remainder = clamp.(1 .- cumulative_p, 0f0, Inf32)
        weight = ifelse.(cumulative_p .+ p .>= recurrent.cfg.act_threshold, remainder, p)
        weight .*= Float32.(still_running)
        h_out .+= reshape(weight, b, t, 1) .* h

        cumulative_p .+= p .* Float32.(still_running)
        halted .|= cumulative_p .>= recurrent.cfg.act_threshold

        if all(halted) && kv_cache === nothing
            break
        end
    end

    return h_out
end
