using Lux
using Random
using Test

function lux_attn_cfg(; attn_type="gqa", kwargs...)
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

@testset "LuxGQAttention parity" begin
    cfg = lux_attn_cfg(attn_type="gqa")
    attn = GQAttention(cfg; rng=MersenneTwister(41))
    layer = LuxGQAttention(cfg)
    ps = to_lux_parameters(attn)
    st = Lux.initialstates(MersenneTwister(1), layer)
    freqs = precompute_rope_freqs(cfg.dim ÷ cfg.n_heads, cfg.max_seq_len)
    x = randn(Float32, 2, 4, cfg.dim)

    expected = attn(x, freqs[1:4, :])
    actual, st_new = Lux.apply(layer, (x=x, freqs_cis=freqs[1:4, :]), ps, st)
    @test st_new == st
    @test actual ≈ expected atol = 1f-6

    legacy_cache = Dict{String, Any}()
    lux_cache = Dict{String, Any}()
    expected_cache = attn(x, freqs[1:4, :]; kv_cache=legacy_cache, cache_key="gq")
    actual_cache, _ = Lux.apply(layer, (x=x, freqs_cis=freqs[1:4, :], kv_cache=lux_cache, cache_key="gq"), ps, st)
    @test actual_cache ≈ expected_cache atol = 1f-6
    @test size(OpenMythos.TransformerCore.buffer_view(legacy_cache["gq"].k)) == size(OpenMythos.TransformerCore.buffer_view(lux_cache["gq"].k))
end

@testset "LuxMLAttention parity" begin
    cfg = lux_attn_cfg(attn_type="mla")
    attn = MLAttention(cfg; rng=MersenneTwister(42))
    layer = LuxMLAttention(cfg)
    ps = to_lux_parameters(attn)
    st = Lux.initialstates(MersenneTwister(1), layer)
    freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
    x = randn(Float32, 2, 4, cfg.dim)

    expected = attn(x, freqs[1:4, :])
    actual, st_new = Lux.apply(layer, (x=x, freqs_cis=freqs[1:4, :]), ps, st)
    @test keys(st_new) == keys(st)
    @test actual ≈ expected atol = 1f-6

    legacy_cache = Dict{String, Any}()
    lux_cache = Dict{String, Any}()
    expected_cache = attn(x, freqs[1:4, :]; kv_cache=legacy_cache, cache_key="mla")
    actual_cache, _ = Lux.apply(layer, (x=x, freqs_cis=freqs[1:4, :], kv_cache=lux_cache, cache_key="mla"), ps, st)
    @test actual_cache ≈ expected_cache atol = 1f-6
    @test size(OpenMythos.TransformerCore.buffer_view(legacy_cache["mla"].c_kv)) == size(OpenMythos.TransformerCore.buffer_view(lux_cache["mla"].c_kv))
    @test size(OpenMythos.TransformerCore.buffer_view(legacy_cache["mla"].k_rope)) == size(OpenMythos.TransformerCore.buffer_view(lux_cache["mla"].k_rope))
end
