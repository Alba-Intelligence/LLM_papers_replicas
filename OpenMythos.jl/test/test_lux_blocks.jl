using Lux
using Random
using Test

function lux_block_cfg(; attn_type="gqa", kwargs...)
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

@testset "LuxTransformerBlock parity" begin
    cfg_gqa = lux_block_cfg(attn_type="gqa")
    freqs_gqa = precompute_rope_freqs(cfg_gqa.dim ÷ cfg_gqa.n_heads, cfg_gqa.max_seq_len)
    block_gqa = TransformerBlock(cfg_gqa; use_moe=false, rng=MersenneTwister(51))
    lux_block_gqa = LuxTransformerBlock(cfg_gqa; use_moe=false)
    ps_gqa = to_lux_parameters(block_gqa)
    st_gqa = Lux.initialstates(MersenneTwister(1), lux_block_gqa)
    x = randn(Float32, 2, 4, cfg_gqa.dim)

    expected_gqa = block_gqa(x, freqs_gqa[1:4, :])
    actual_gqa, st_gqa_new = Lux.apply(lux_block_gqa, (x=x, freqs_cis=freqs_gqa[1:4, :]), ps_gqa, st_gqa)
    @test keys(st_gqa_new) == keys(st_gqa)
    @test actual_gqa ≈ expected_gqa atol = 1f-6

    cache_legacy = Dict{String, Any}()
    cache_lux = Dict{String, Any}()
    expected_cache = block_gqa(x, freqs_gqa[1:4, :]; kv_cache=cache_legacy, cache_key="blk")
    actual_cache, _ = Lux.apply(lux_block_gqa, (x=x, freqs_cis=freqs_gqa[1:4, :], kv_cache=cache_lux, cache_key="blk"), ps_gqa, st_gqa)
    @test actual_cache ≈ expected_cache atol = 1f-6
    @test size(OpenMythos.TransformerCore.buffer_view(cache_legacy["blk"].k)) == size(OpenMythos.TransformerCore.buffer_view(cache_lux["blk"].k))

    cfg_mla = lux_block_cfg(attn_type="mla")
    freqs_mla = precompute_rope_freqs(cfg_mla.qk_rope_head_dim, cfg_mla.max_seq_len)
    block_mla = TransformerBlock(cfg_mla; use_moe=false, rng=MersenneTwister(52))
    lux_block_mla = LuxTransformerBlock(cfg_mla; use_moe=false)
    ps_mla = to_lux_parameters(block_mla)
    st_mla = Lux.initialstates(MersenneTwister(1), lux_block_mla)

    expected_mla = block_mla(x, freqs_mla[1:4, :])
    actual_mla, _ = Lux.apply(lux_block_mla, (x=x, freqs_cis=freqs_mla[1:4, :]), ps_mla, st_mla)
    @test actual_mla ≈ expected_mla atol = 1f-6
end

@testset "LuxRecurrentBlock parity" begin
    cfg = lux_block_cfg(attn_type="gqa")
    freqs = precompute_rope_freqs(cfg.dim ÷ cfg.n_heads, cfg.max_seq_len)
    recurrent = RecurrentBlock(cfg; rng=MersenneTwister(53))
    lux_recurrent = LuxRecurrentBlock(cfg)
    ps = to_lux_parameters(recurrent)
    st = Lux.initialstates(MersenneTwister(1), lux_recurrent)

    h = randn(Float32, 2, 4, cfg.dim)
    e = randn(Float32, 2, 4, cfg.dim)

    expected = recurrent(copy(h), copy(e), freqs[1:4, :]; n_loops=3)
    actual, st_new = Lux.apply(lux_recurrent, (h=h, e=e, freqs_cis=freqs[1:4, :], n_loops=3), ps, st)
    @test keys(st_new) == keys(st)
    @test actual ≈ expected atol = 1f-6

    shallow_expected = recurrent(copy(h), copy(e), freqs[1:4, :]; n_loops=1)
    shallow_actual, _ = Lux.apply(lux_recurrent, (h=h, e=e, freqs_cis=freqs[1:4, :], n_loops=1), ps, st)
    @test shallow_actual ≈ shallow_expected atol = 1f-6
end
