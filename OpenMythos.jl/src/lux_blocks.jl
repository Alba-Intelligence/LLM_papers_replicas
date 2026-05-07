_norm_parameters(norm::RMSNorm) = (weight=copy(norm.weight),)

"""
    LuxTransformerBlock

Lux-native transformer block matching `TransformerBlock`.
"""
struct LuxTransformerBlock{AN,FN} <: Lux.LuxCore.AbstractLuxLayer
    attn_norm::RMSNorm{Float32}
    ffn_norm::RMSNorm{Float32}
    attn::AN
    ffn::FN
end

function LuxTransformerBlock(cfg::MythosConfig; use_moe::Bool=false)
    attn = cfg.attn_type == "mla" ? LuxMLAttention(cfg) : LuxGQAttention(cfg)
    ffn = use_moe ? LuxMoEFFN(cfg) : LuxExpert(cfg.dim, cfg.dim * 4 ÷ 3)
    return LuxTransformerBlock(RMSNorm(cfg.dim), RMSNorm(cfg.dim), attn, ffn)
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxTransformerBlock)
    return (
        attn_norm=Lux.initialparameters(rng, layer.attn_norm),
        ffn_norm=Lux.initialparameters(rng, layer.ffn_norm),
        attn=Lux.initialparameters(rng, layer.attn),
        ffn=Lux.initialparameters(rng, layer.ffn),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::LuxTransformerBlock)
    return (
        attn_norm=Lux.initialstates(rng, layer.attn_norm),
        ffn_norm=Lux.initialstates(rng, layer.ffn_norm),
        attn=Lux.initialstates(rng, layer.attn),
        ffn=Lux.initialstates(rng, layer.ffn),
    )
end

function (layer::LuxTransformerBlock)(input::NamedTuple, ps, st)
    x = input.x
    freqs_cis = input.freqs_cis
    mask = get(input, :mask, nothing)
    kv_cache = get(input, :kv_cache, nothing)
    kv_capacity = get(input, :kv_capacity, nothing)
    cache_key = get(input, :cache_key, "default")

    attn_in, st_attn_norm = Lux.apply(layer.attn_norm, x, ps.attn_norm, st.attn_norm)
    attn_out, st_attn = Lux.apply(
        layer.attn,
        (x=attn_in, freqs_cis=freqs_cis, mask=mask, kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key=cache_key),
        ps.attn,
        st.attn,
    )
    x = x .+ attn_out

    ffn_in, st_ffn_norm = Lux.apply(layer.ffn_norm, x, ps.ffn_norm, st.ffn_norm)
    ffn_out, st_ffn = Lux.apply(layer.ffn, ffn_in, ps.ffn, st.ffn)
    x = x .+ ffn_out

    return x, (attn_norm=st_attn_norm, ffn_norm=st_ffn_norm, attn=st_attn, ffn=st_ffn)
end

function _block_parameters(block::TransformerBlock)
    return (
        attn_norm=_norm_parameters(block.attn_norm),
        ffn_norm=_norm_parameters(block.ffn_norm),
        attn=to_lux_parameters(block.attn),
        ffn=to_lux_parameters(block.ffn),
    )
end

"""
    LuxRecurrentBlock

Lux-native recurrent block matching `RecurrentBlock`.
"""
struct LuxRecurrentBlock <: Lux.LuxCore.AbstractLuxLayer
    cfg::MythosConfig
    block::LuxTransformerBlock
    injection::LuxLTIInjection
    act::LuxACTHalting
    lora::LuxLoRAAdapter
    norm::RMSNorm{Float32}
    loop_dim::Int
end

function LuxRecurrentBlock(cfg::MythosConfig)
    return LuxRecurrentBlock(
        cfg,
        LuxTransformerBlock(cfg; use_moe=true),
        LuxLTIInjection(cfg.dim),
        LuxACTHalting(cfg.dim),
        LuxLoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters),
        RMSNorm(cfg.dim),
        cfg.dim ÷ 8,
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxRecurrentBlock)
    return (
        block=Lux.initialparameters(rng, layer.block),
        injection=Lux.initialparameters(rng, layer.injection),
        act=Lux.initialparameters(rng, layer.act),
        lora=Lux.initialparameters(rng, layer.lora),
        norm=Lux.initialparameters(rng, layer.norm),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::LuxRecurrentBlock)
    return (
        block=Lux.initialstates(rng, layer.block),
        injection=Lux.initialstates(rng, layer.injection),
        act=Lux.initialstates(rng, layer.act),
        lora=Lux.initialstates(rng, layer.lora),
        norm=Lux.initialstates(rng, layer.norm),
    )
end

function (layer::LuxRecurrentBlock)(input::NamedTuple, ps, st)
    h = input.h
    e = input.e
    freqs_cis = input.freqs_cis
    mask = get(input, :mask, nothing)
    n_loops = get(input, :n_loops, nothing)
    kv_cache = get(input, :kv_cache, nothing)
    kv_capacity = get(input, :kv_capacity, nothing)

    loops = something(n_loops, layer.cfg.max_loop_iters)
    b, t, d = size(h)
    halted = falses(b, t)
    cumulative_p = zeros(Float32, b, t)
    h_out = zeros(eltype(h), b, t, d)

    st_block = st.block
    st_injection = st.injection
    st_act = st.act
    st_lora = st.lora
    st_norm = st.norm

    for loop_t in 0:(loops - 1)
        h_loop = loop_index_embedding(h, loop_t, layer.loop_dim)
        combined, st_norm = Lux.apply(layer.norm, h_loop .+ e, ps.norm, st_norm)
        cache_key = "recurrent_loop_$(loop_t)"
        trans_out, st_block = Lux.apply(
            layer.block,
            (x=combined, freqs_cis=freqs_cis, mask=mask, kv_cache=kv_cache, kv_capacity=kv_capacity, cache_key=cache_key),
            ps.block,
            st_block,
        )
        lora_out, st_lora = Lux.apply(layer.lora, (trans_out, loop_t), ps.lora, st_lora)
        trans_out = trans_out .+ lora_out
        h, st_injection = Lux.apply(layer.injection, (h, e, trans_out), ps.injection, st_injection)

        p, st_act = Lux.apply(layer.act, h, ps.act, st_act)
        still_running = .!halted
        remainder = clamp.(1 .- cumulative_p, 0f0, Inf32)
        weight = ifelse.(cumulative_p .+ p .>= layer.cfg.act_threshold, remainder, p)
        weight = weight .* Float32.(still_running)
        h_out = h_out .+ reshape(weight, b, t, 1) .* h

        cumulative_p = cumulative_p .+ p .* Float32.(still_running)
        halted = halted .| (cumulative_p .>= layer.cfg.act_threshold)

        if all(halted) && kv_cache === nothing
            break
        end
    end

    return h_out, (block=st_block, injection=st_injection, act=st_act, lora=st_lora, norm=st_norm)
end

function _recurrent_parameters(recurrent::RecurrentBlock)
    return (
        block=_block_parameters(recurrent.block),
        injection=to_lux_parameters(recurrent.injection),
        act=to_lux_parameters(recurrent.act),
        lora=to_lux_parameters(recurrent.lora),
        norm=_norm_parameters(recurrent.norm),
    )
end

to_lux_parameters(block::TransformerBlock) = _block_parameters(block)
to_lux_parameters(recurrent::RecurrentBlock) = _recurrent_parameters(recurrent)
