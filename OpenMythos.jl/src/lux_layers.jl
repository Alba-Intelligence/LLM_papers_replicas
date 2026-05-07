const _ROUTED_EXPERT_KEYS = n -> Tuple(Symbol(:routed_expert_, i) for i in 1:n)
const _SHARED_EXPERT_KEYS = n -> Tuple(Symbol(:shared_expert_, i) for i in 1:n)

"""
    LuxExpert

Lux-native SwiGLU expert matching the OpenMythos `Expert` math.
"""
struct LuxExpert{G,U,D} <: Lux.LuxCore.AbstractLuxLayer
    gate::G
    up::U
    down::D
end

function LuxExpert(dim::Integer, expert_dim::Integer; T::Type{<:AbstractFloat}=Float32)
    return LuxExpert(
        TransformerCore.FeatureLinear(dim, expert_dim; bias=false, T=T),
        TransformerCore.FeatureLinear(dim, expert_dim; bias=false, T=T),
        TransformerCore.FeatureLinear(expert_dim, dim; bias=false, T=T),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxExpert)
    return (
        gate=Lux.initialparameters(rng, layer.gate),
        up=Lux.initialparameters(rng, layer.up),
        down=Lux.initialparameters(rng, layer.down),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::LuxExpert)
    return (
        gate=Lux.initialstates(rng, layer.gate),
        up=Lux.initialstates(rng, layer.up),
        down=Lux.initialstates(rng, layer.down),
    )
end

function (layer::LuxExpert)(x::AbstractArray, ps, st)
    gate, st_gate = Lux.apply(layer.gate, x, ps.gate, st.gate)
    up, st_up = Lux.apply(layer.up, x, ps.up, st.up)
    out, st_down = Lux.apply(layer.down, _silu.(gate) .* up, ps.down, st.down)
    return out, (gate=st_gate, up=st_up, down=st_down)
end

"""Apply a `LuxExpert` to a single feature vector."""
function (layer::LuxExpert)(x::AbstractVector, ps, st)
    y, st_new = layer(reshape(x, 1, 1, :), ps, st)
    return vec(dropdims(y; dims=(1, 2))), st_new
end

_expert_namedtuple(keys::Tuple, values::Tuple) = NamedTuple{keys}(values)

function _expert_parameters(expert::Expert)
    return (
        gate=(weight=copy(expert.gate),),
        up=(weight=copy(expert.up),),
        down=(weight=copy(expert.down),),
    )
end

"""
    LuxMoEFFN

Lux-native mixture-of-experts feed-forward network matching `MoEFFN`.
"""
struct LuxMoEFFN <: Lux.LuxCore.AbstractLuxLayer
    n_experts::Int
    n_shared::Int
    topk::Int
    dim::Int
    expert_dim::Int
    routed_expert_dim::Int
    routed_layers::Tuple
    shared_layers::Tuple
end

function LuxMoEFFN(cfg::MythosConfig; T::Type{<:AbstractFloat}=Float32)
    routed_layers = ntuple(_ -> LuxExpert(cfg.dim, cfg.expert_dim; T=T), cfg.n_experts)
    shared_layers = Tuple(LuxExpert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok; T=T) for _ in 1:cfg.n_shared_experts)
    return LuxMoEFFN(
        cfg.n_experts,
        cfg.n_shared_experts,
        cfg.n_experts_per_tok,
        cfg.dim,
        cfg.expert_dim,
        cfg.expert_dim * cfg.n_experts_per_tok,
        routed_layers,
        shared_layers,
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxMoEFFN)
    routed_keys = _ROUTED_EXPERT_KEYS(layer.n_experts)
    shared_keys = _SHARED_EXPERT_KEYS(layer.n_shared)
    routed = _expert_namedtuple(routed_keys, ntuple(i -> Lux.initialparameters(rng, layer.routed_layers[i]), layer.n_experts))
    shared = _expert_namedtuple(shared_keys, ntuple(i -> Lux.initialparameters(rng, layer.shared_layers[i]), layer.n_shared))
    return (
        router=(weight=Float32.(0.02 .* randn(rng, layer.n_experts, layer.dim)), bias=zeros(Float32, layer.n_experts)),
        routed_experts=routed,
        shared_experts=shared,
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::LuxMoEFFN)
    routed_keys = _ROUTED_EXPERT_KEYS(layer.n_experts)
    shared_keys = _SHARED_EXPERT_KEYS(layer.n_shared)
    routed = _expert_namedtuple(routed_keys, ntuple(i -> Lux.initialstates(rng, layer.routed_layers[i]), layer.n_experts))
    shared = _expert_namedtuple(shared_keys, ntuple(i -> Lux.initialstates(rng, layer.shared_layers[i]), layer.n_shared))
    return (routed_experts=routed, shared_experts=shared)
end

function _router_logits(ps, token_matrix)
    return ps.router.weight * transpose(token_matrix)
end

function _shared_moe_state(layer::LuxMoEFFN, st, routed_states, shared_states)
    routed = _expert_namedtuple(_ROUTED_EXPERT_KEYS(layer.n_experts), Tuple(routed_states))
    shared = _expert_namedtuple(_SHARED_EXPERT_KEYS(layer.n_shared), Tuple(shared_states))
    return (routed_experts=routed, shared_experts=shared)
end

function (layer::LuxMoEFFN)(x::AbstractArray{T, 3}, ps, st) where {T<:AbstractFloat}
    if layer.n_experts == 1 && layer.topk == 1
        out, routed_st = Lux.apply(layer.routed_layers[1], x, getfield(ps.routed_experts, _ROUTED_EXPERT_KEYS(layer.n_experts)[1]), getfield(st.routed_experts, _ROUTED_EXPERT_KEYS(layer.n_experts)[1]))
        shared_states = Any[]
        for i in 1:layer.n_shared
            key = _SHARED_EXPERT_KEYS(layer.n_shared)[i]
            add, shared_st = Lux.apply(layer.shared_layers[i], x, getfield(ps.shared_experts, key), getfield(st.shared_experts, key))
            out = out .+ add
            push!(shared_states, shared_st)
        end
        routed_states = Any[routed_st]
        return out, _shared_moe_state(layer, st, routed_states, shared_states)
    end

    b, t, d = size(x)
    token_matrix = reshape(permutedims(x, (3, 1, 2)), d, :)'
    n = size(token_matrix, 1)

    logits = _router_logits(ps, token_matrix)
    scores = _softmax_cols(logits)

    routed_states = Any[getfield(st.routed_experts, key) for key in _ROUTED_EXPERT_KEYS(layer.n_experts)]
    token_outputs = [
        begin
            adjusted = view(logits, :, token_idx) .+ ps.router.bias
            top_idx = partialsortperm(vec(adjusted), 1:layer.topk; rev=true)
            token_scores = scores[top_idx, token_idx]
            token_scores = token_scores ./ sum(token_scores)
            token = vec(@view token_matrix[token_idx, :])
            reduce(
                .+,
                (
                    begin
                        key = _ROUTED_EXPERT_KEYS(layer.n_experts)[expert_idx]
                        expert_out, routed_st = Lux.apply(layer.routed_layers[expert_idx], token, getfield(ps.routed_experts, key), routed_states[expert_idx])
                        routed_states[expert_idx] = routed_st
                        score .* expert_out
                    end
                    for (score, expert_idx) in zip(token_scores, top_idx)
                );
                init=zeros(T, d),
            )
        end
        for token_idx in 1:n
    ]
    out = cat([reshape(token_out, 1, d) for token_out in token_outputs]...; dims=1)

    shared_states = Any[]
    for i in 1:layer.n_shared
        key = _SHARED_EXPERT_KEYS(layer.n_shared)[i]
        shared_out, shared_st = Lux.apply(layer.shared_layers[i], token_matrix, getfield(ps.shared_experts, key), getfield(st.shared_experts, key))
        out = out .+ shared_out
        push!(shared_states, shared_st)
    end

    return permutedims(reshape(out, b, t, d), (1, 2, 3)), _shared_moe_state(layer, st, routed_states, shared_states)
end

function _moe_parameters(moe::MoEFFN)
    routed_keys = _ROUTED_EXPERT_KEYS(moe.n_experts)
    shared_keys = _SHARED_EXPERT_KEYS(moe.n_shared)
    routed = _expert_namedtuple(routed_keys, Tuple(_expert_parameters(expert) for expert in moe.routed_experts))
    shared = _expert_namedtuple(shared_keys, Tuple(_expert_parameters(expert) for expert in moe.shared_experts))
    return (
        router=(weight=copy(moe.router), bias=copy(moe.router_bias)),
        routed_experts=routed,
        shared_experts=shared,
    )
end

"""
    LuxLoRAAdapter

Lux-native depth-wise LoRA adapter matching `LoRAAdapter`.
"""
struct LuxLoRAAdapter <: Lux.LuxCore.AbstractLuxLayer
    dim::Int
    rank::Int
    max_loops::Int
    init_scale::Float32
end

function LuxLoRAAdapter(dim::Integer, rank::Integer, max_loops::Integer; init_scale::Real=0.02f0)
    dim > 0 || throw(ArgumentError("dim must be positive"))
    rank > 0 || throw(ArgumentError("rank must be positive"))
    max_loops > 0 || throw(ArgumentError("max_loops must be positive"))
    return LuxLoRAAdapter(Int(dim), Int(rank), Int(max_loops), Float32(init_scale))
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxLoRAAdapter)
    return (
        down=Float32.(layer.init_scale .* randn(rng, layer.rank, layer.dim)),
        B=Float32.(layer.init_scale .* randn(rng, layer.dim, layer.rank)),
        scale=Float32.(layer.init_scale .* randn(rng, layer.rank, layer.max_loops)),
    )
end

Lux.initialstates(::AbstractRNG, ::LuxLoRAAdapter) = NamedTuple()

function (layer::LuxLoRAAdapter)(input::Tuple{<:AbstractArray, <:Integer}, ps, st)
    x, loop_t = input
    t_idx = clamp(loop_t + 1, 1, size(ps.scale, 2))
    flat = _flatten_feature_last(x)
    down = ps.down * flat
    scaled = down .* reshape(ps.scale[:, t_idx], :, 1)
    out = ps.B * scaled
    return _unflatten_feature_last(out, size(x)), st
end

_lora_parameters(adapter::LoRAAdapter) = (down=copy(adapter.down), B=copy(adapter.B), scale=copy(adapter.scale))

"""
    LuxLTIInjection

Lux-native stable recurrent injection matching `LTIInjection`.
"""
struct LuxLTIInjection <: Lux.LuxCore.AbstractLuxLayer
    dim::Int
end

LuxLTIInjection(dim::Integer) = (dim > 0 || throw(ArgumentError("dim must be positive")); LuxLTIInjection(Int(dim)))

function Lux.initialparameters(::AbstractRNG, layer::LuxLTIInjection)
    return (log_A=zeros(Float32, layer.dim), log_dt=zeros(Float32, 1), B=fill(0.1f0, layer.dim))
end

Lux.initialstates(::AbstractRNG, ::LuxLTIInjection) = NamedTuple()

function _get_A_from_ps(ps)
    raw = exp.(-exp.(clamp.(ps.log_dt .+ ps.log_A, -20, 20)))
    return max.(raw, floatmin(eltype(raw)))
end

function (layer::LuxLTIInjection)(input::Tuple{<:AbstractArray, <:AbstractArray, <:AbstractArray}, ps, st)
    h, e, transformer_out = input
    size(h) == size(e) == size(transformer_out) || throw(DimensionMismatch("all inputs must share shape"))
    A = _feature_broadcast(_get_A_from_ps(ps), ndims(h))
    B = _feature_broadcast(ps.B, ndims(h))
    return A .* h .+ B .* e .+ transformer_out, st
end

_lti_parameters(inj::LTIInjection) = (log_A=copy(inj.log_A), log_dt=copy(inj.log_dt), B=copy(inj.B))

"""
    LuxACTHalting

Lux-native ACT halting head matching `ACTHalting`.
"""
struct LuxACTHalting{P} <: Lux.LuxCore.AbstractLuxLayer
    proj::P
end

LuxACTHalting(dim::Integer; T::Type{<:AbstractFloat}=Float32) = LuxACTHalting(TransformerCore.FeatureLinear(dim, 1; bias=true, T=T))

function Lux.initialparameters(rng::AbstractRNG, layer::LuxACTHalting)
    return (proj=Lux.initialparameters(rng, layer.proj),)
end

function Lux.initialstates(rng::AbstractRNG, layer::LuxACTHalting)
    return (proj=Lux.initialstates(rng, layer.proj),)
end

function (layer::LuxACTHalting)(h::AbstractArray, ps, st)
    logits, st_proj = Lux.apply(layer.proj, h, ps.proj, st.proj)
    return dropdims(_sigmoid.(logits); dims=ndims(logits)), (proj=st_proj,)
end

_act_parameters(act::ACTHalting) = (proj=(weight=copy(act.weight), bias=copy(act.bias)),)

"""
    to_lux_parameters(obj)

Return a Lux-style parameter tree initialized from an existing OpenMythos leaf
module.
"""
to_lux_parameters(expert::Expert) = _expert_parameters(expert)
to_lux_parameters(moe::MoEFFN) = _moe_parameters(moe)
to_lux_parameters(adapter::LoRAAdapter) = _lora_parameters(adapter)
to_lux_parameters(inj::LTIInjection) = _lti_parameters(inj)
to_lux_parameters(act::ACTHalting) = _act_parameters(act)
