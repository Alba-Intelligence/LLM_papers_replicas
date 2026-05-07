using Lux
using Random
using Test

function lux_cfg(; kwargs...)
    return MythosConfig(
        vocab_size=128,
        dim=16,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        max_loop_iters=5,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=8,
        lora_rank=4,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4;
        kwargs...,
    )
end

@testset "LuxExpert parity" begin
    rng = MersenneTwister(31)
    expert = Expert(16, 8; rng=rng)
    layer = LuxExpert(16, 8)
    ps = to_lux_parameters(expert)
    st = Lux.initialstates(MersenneTwister(1), layer)

    x = randn(Float32, 2, 3, 16)
    expected = expert(x)
    actual, st_new = Lux.apply(layer, x, ps, st)
    @test st_new == st
    @test actual ≈ expected atol = 1f-6

    v = randn(Float32, 16)
    expected_v = expert(v)
    actual_v, _ = Lux.apply(layer, v, ps, st)
    @test actual_v ≈ expected_v atol = 1f-6
end

@testset "LuxMoEFFN parity" begin
    cfg = lux_cfg()
    moe = MoEFFN(cfg; rng=MersenneTwister(32))
    layer = LuxMoEFFN(cfg)
    ps = to_lux_parameters(moe)
    st = Lux.initialstates(MersenneTwister(1), layer)

    x = randn(Float32, 2, 3, cfg.dim)
    expected = moe(x)
    actual, st_new = Lux.apply(layer, x, ps, st)
    @test keys(st_new.routed_experts) == keys(st.routed_experts)
    @test keys(st_new.shared_experts) == keys(st.shared_experts)
    @test actual ≈ expected atol = 1f-6
end

@testset "Lux recurrent leaf parity" begin
    x = randn(Float32, 2, 3, 16)

    lora = LoRAAdapter(16, 4, 5; rng=MersenneTwister(33))
    lux_lora = LuxLoRAAdapter(16, 4, 5)
    lora_ps = to_lux_parameters(lora)
    lora_st = Lux.initialstates(MersenneTwister(1), lux_lora)
    expected_lora = lora(x, 2)
    actual_lora, lora_st_new = Lux.apply(lux_lora, (x, 2), lora_ps, lora_st)
    @test lora_st_new == lora_st
    @test actual_lora ≈ expected_lora atol = 1f-6

    inj = LTIInjection(16)
    lux_inj = LuxLTIInjection(16)
    inj_ps = to_lux_parameters(inj)
    inj_st = Lux.initialstates(MersenneTwister(1), lux_inj)
    h = randn(Float32, 2, 3, 16)
    e = randn(Float32, 2, 3, 16)
    t = randn(Float32, 2, 3, 16)
    expected_inj = inj(h, e, t)
    actual_inj, inj_st_new = Lux.apply(lux_inj, (h, e, t), inj_ps, inj_st)
    @test inj_st_new == inj_st
    @test actual_inj ≈ expected_inj atol = 1f-6

    act = ACTHalting(16; rng=MersenneTwister(34))
    lux_act = LuxACTHalting(16)
    act_ps = to_lux_parameters(act)
    act_st = Lux.initialstates(MersenneTwister(1), lux_act)
    expected_act = act(x)
    actual_act, act_st_new = Lux.apply(lux_act, x, act_ps, act_st)
    @test act_st_new == act_st
    @test actual_act ≈ expected_act atol = 1f-6
end
