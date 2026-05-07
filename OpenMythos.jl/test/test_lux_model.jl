using Lux
using Random
using Test

function lux_model_cfg(; attn_type="gqa", kwargs...)
    return MythosConfig(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type=attn_type,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8;
        kwargs...,
    )
end

@testset "LuxOpenMythos parity" begin
    for attn_type in ("gqa", "mla")
        cfg = lux_model_cfg(attn_type=attn_type)
        model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(attn_type == "gqa" ? 61 : 62))
        layer = LuxOpenMythos(cfg)
        ps = to_lux_parameters(model)
        st = Lux.initialstates(MersenneTwister(1), layer)

        ids = rand(MersenneTwister(7), 0:(cfg.vocab_size - 1), 2, 4)
        expected = model(ids; n_loops=2)
        actual, st_new = Lux.apply(layer, (input_ids=ids, n_loops=2), ps, st)
        @test keys(st_new) == keys(st)
        @test actual ≈ expected atol = 1f-6
        @test !(:head in keys(ps))
        @test size(ps.embed.weight) == (cfg.vocab_size, cfg.dim)

        legacy_cache = Dict{String, Any}()
        lux_cache = Dict{String, Any}()
        expected_cache = model(ids; n_loops=2, kv_cache=legacy_cache)
        actual_cache, _ = Lux.apply(layer, (input_ids=ids, n_loops=2, kv_cache=lux_cache), ps, st)
        @test actual_cache ≈ expected_cache atol = 1f-6
        @test keys(lux_cache) == keys(legacy_cache)
    end
end

@testset "LuxOpenMythos tied embedding head behavior" begin
    cfg = lux_model_cfg(attn_type="gqa", vocab_size=32, dim=16, n_heads=4, n_kv_heads=2, expert_dim=8, kv_lora_rank=8, q_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4)
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(63))
    layer = LuxOpenMythos(cfg)
    ps = to_lux_parameters(model)
    st = Lux.initialstates(MersenneTwister(1), layer)
    ids = reshape(Int[1, 2, 3, 4], 1, :)

    logits_before, _ = Lux.apply(layer, ids, ps, st)
    ps_shifted = merge(ps, (embed=(weight=ps.embed.weight .+ 0.01f0,),))
    logits_after, _ = Lux.apply(layer, ids, ps_shifted, st)
    @test !all(isapprox.(logits_before, logits_after; atol=1f-6))
end
