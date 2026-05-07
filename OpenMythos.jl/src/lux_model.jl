const _PRELUDE_KEYS = n -> Tuple(Symbol(:prelude_, i) for i in 1:n)
const _CODA_KEYS = n -> Tuple(Symbol(:coda_, i) for i in 1:n)

_stack_namedtuple(keys::Tuple, values::Tuple) = NamedTuple{keys}(values)

"""
    LuxOpenMythos

Lux-native tied-embedding OpenMythos model mirror. It matches the current
manual `OpenMythos` forward path while representing trainable arrays as a Lux
parameter tree.
"""
struct LuxOpenMythos{E,P,R,C,N} <: Lux.LuxCore.AbstractLuxLayer
    cfg::MythosConfig
    freqs_cis::Matrix{ComplexF32}
    freqs_cis_mla::Matrix{ComplexF32}
    embed::E
    prelude::P
    recurrent::R
    coda::C
    norm::N
end

function LuxOpenMythos(cfg::MythosConfig)
    prelude = Tuple(LuxTransformerBlock(cfg; use_moe=false) for _ in 1:cfg.prelude_layers)
    coda = Tuple(LuxTransformerBlock(cfg; use_moe=false) for _ in 1:cfg.coda_layers)
    return LuxOpenMythos(
        cfg,
        precompute_rope_freqs(cfg.dim ÷ cfg.n_heads, cfg.max_seq_len; theta=cfg.rope_theta),
        precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len; theta=cfg.rope_theta),
        TransformerCore.TokenEmbedding(cfg.vocab_size, cfg.dim),
        prelude,
        LuxRecurrentBlock(cfg),
        coda,
        RMSNorm(cfg.dim),
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxOpenMythos)
    prelude = _stack_namedtuple(_PRELUDE_KEYS(length(layer.prelude)), ntuple(i -> Lux.initialparameters(rng, layer.prelude[i]), length(layer.prelude)))
    coda = _stack_namedtuple(_CODA_KEYS(length(layer.coda)), ntuple(i -> Lux.initialparameters(rng, layer.coda[i]), length(layer.coda)))
    return (
        embed=Lux.initialparameters(rng, layer.embed),
        prelude=prelude,
        recurrent=Lux.initialparameters(rng, layer.recurrent),
        coda=coda,
        norm=Lux.initialparameters(rng, layer.norm),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::LuxOpenMythos)
    prelude = _stack_namedtuple(_PRELUDE_KEYS(length(layer.prelude)), ntuple(i -> Lux.initialstates(rng, layer.prelude[i]), length(layer.prelude)))
    coda = _stack_namedtuple(_CODA_KEYS(length(layer.coda)), ntuple(i -> Lux.initialstates(rng, layer.coda[i]), length(layer.coda)))
    return (
        embed=Lux.initialstates(rng, layer.embed),
        prelude=prelude,
        recurrent=Lux.initialstates(rng, layer.recurrent),
        coda=coda,
        norm=Lux.initialstates(rng, layer.norm),
    )
end

function _lux_forward_hidden(
    layer::LuxOpenMythos,
    input_ids::AbstractMatrix{<:Integer},
    ps,
    st;
    n_loops::Union{Nothing, Integer}=nothing,
    kv_cache::Union{Nothing, AbstractDict}=nothing,
    start_pos::Integer=0,
    kv_capacity::Union{Nothing, Integer}=nothing,
)
    t = size(input_ids, 2)
    x, st_embed = Lux.apply(layer.embed, input_ids, ps.embed, st.embed)
    freqs_all = layer.cfg.attn_type == "mla" ? layer.freqs_cis_mla : layer.freqs_cis
    freqs = freqs_all[(start_pos + 1):(start_pos + t), :]
    mask = t > 1 ? _causal_mask(t, start_pos, Float32) : nothing

    for i in eachindex(layer.prelude)
        key = _PRELUDE_KEYS(length(layer.prelude))[i]
        x, _ = Lux.apply(
            layer.prelude[i],
            (x=x, freqs_cis=freqs, mask=mask, kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key="prelude_$(i - 1)"),
            getfield(ps.prelude, key),
            getfield(st.prelude, key),
        )
    end

    e = x
    x, _ = Lux.apply(
        layer.recurrent,
        (h=x, e=e, freqs_cis=freqs, mask=mask, n_loops=n_loops, kv_cache=kv_cache, kv_capacity=kv_capacity),
        ps.recurrent,
        st.recurrent,
    )

    for i in eachindex(layer.coda)
        key = _CODA_KEYS(length(layer.coda))[i]
        x, _ = Lux.apply(
            layer.coda[i],
            (x=x, freqs_cis=freqs, mask=mask, kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key="coda_$(i - 1)"),
            getfield(ps.coda, key),
            getfield(st.coda, key),
        )
    end

    x, _ = Lux.apply(layer.norm, x, ps.norm, st.norm)
    return x, st
end

function (layer::LuxOpenMythos)(input_ids::AbstractMatrix{<:Integer}, ps, st)
    hidden, st_new = _lux_forward_hidden(layer, input_ids, ps, st)
    return TransformerCore.tied_lm_head(hidden, ps.embed.weight), st_new
end

function (layer::LuxOpenMythos)(input::NamedTuple, ps, st)
    input_ids = input.input_ids
    hidden, st_new = _lux_forward_hidden(
        layer,
        input_ids,
        ps,
        st;
        n_loops=get(input, :n_loops, nothing),
        kv_cache=get(input, :kv_cache, nothing),
        start_pos=get(input, :start_pos, 0),
        kv_capacity=get(input, :kv_capacity, nothing),
    )
    return TransformerCore.tied_lm_head(hidden, ps.embed.weight), st_new
end

function _stack_parameters(keys::Tuple, values::Tuple)
    return _stack_namedtuple(keys, values)
end

function _model_parameters(model::OpenMythos)
    prelude = _stack_parameters(_PRELUDE_KEYS(length(model.prelude)), Tuple(to_lux_parameters(layer) for layer in model.prelude))
    coda = _stack_parameters(_CODA_KEYS(length(model.coda)), Tuple(to_lux_parameters(layer) for layer in model.coda))
    return (
        embed=(weight=copy(model.embed),),
        prelude=prelude,
        recurrent=to_lux_parameters(model.recurrent),
        coda=coda,
        norm=_norm_parameters(model.norm),
    )
end

to_lux_parameters(model::OpenMythos) = _model_parameters(model)
