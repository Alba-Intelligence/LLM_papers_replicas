using Test
using Random

const B = 2
const T = 8

function gqa_cfg(; kwargs...)
    base = (
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99f0,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    return MythosConfig(; base..., kwargs...)
end

mla_cfg(; kwargs...) = gqa_cfg(; attn_type="mla", kwargs...)

@testset "GQAttention" begin
    Random.seed!(1)
    cfg = gqa_cfg()
    freqs = precompute_rope_freqs(cfg.dim ÷ cfg.n_heads, cfg.max_seq_len)
    attn = GQAttention(cfg)
    x = randn(Float32, B, T, cfg.dim)

    out = attn(x, freqs[1:T, :])
    @test size(out) == (B, T, cfg.dim)

    cache = Dict{String, Any}()
    attn(x, freqs[1:T, :]; kv_cache=cache, cache_key="layer0")
    first_len = size(cache["layer0"]["k"], 2)
    attn(x, freqs[1:T, :]; kv_cache=cache, cache_key="layer0")
    @test size(cache["layer0"]["k"], 2) == first_len + T

    mask = zeros(Float32, 1, 1, T, T)
    for i in 1:T, j in (i + 1):T
        mask[1, 1, i, j] = -Inf32
    end
    @test size(attn(x, freqs[1:T, :]; mask=mask)) == (B, T, cfg.dim)
end

@testset "MLAttention" begin
    Random.seed!(2)
    cfg = mla_cfg()
    freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
    attn = MLAttention(cfg)
    x = randn(Float32, B, T, cfg.dim)

    out = attn(x, freqs[1:T, :])
    @test size(out) == (B, T, cfg.dim)

    cache = Dict{String, Any}()
    attn(x, freqs[1:T, :]; kv_cache=cache, cache_key="mla0")
    @test haskey(cache["mla0"], "c_kv")
    @test haskey(cache["mla0"], "k_rope")
    @test size(cache["mla0"]["c_kv"], 3) == cfg.kv_lora_rank

    first_len = size(cache["mla0"]["c_kv"], 2)
    attn(x, freqs[1:T, :]; kv_cache=cache, cache_key="mla0")
    @test size(cache["mla0"]["c_kv"], 2) == first_len + T
end

@testset "Expert and MoEFFN" begin
    Random.seed!(3)
    expert = Expert(64, 32)
    x3 = randn(Float32, B, T, 64)
    @test size(expert(x3)) == (B, T, 64)
    @test size(expert(randn(Float32, 64))) == (64,)

    cfg = gqa_cfg()
    moe = MoEFFN(cfg)
    out = moe(randn(Float32, B, T, cfg.dim))
    @test size(out) == (B, T, cfg.dim)
    @test length(moe.router_bias) == cfg.n_experts

    for exp in moe.routed_experts
        exp.gate .= 0
        exp.up .= 0
        exp.down .= 0
    end
    shared_out = moe(randn(Float32, B, T, cfg.dim))
    @test sum(abs, shared_out) > 0
end

@testset "TransformerBlock and RecurrentBlock" begin
    Random.seed!(4)
    cfg_gqa = gqa_cfg()
    freqs_gqa = precompute_rope_freqs(cfg_gqa.dim ÷ cfg_gqa.n_heads, cfg_gqa.max_seq_len)
    block_gqa = TransformerBlock(cfg_gqa; use_moe=false)
    x = randn(Float32, B, T, cfg_gqa.dim)
    @test size(block_gqa(x, freqs_gqa[1:T, :])) == (B, T, cfg_gqa.dim)
    @test block_gqa.attn isa GQAttention

    cfg_mla = mla_cfg()
    freqs_mla = precompute_rope_freqs(cfg_mla.qk_rope_head_dim, cfg_mla.max_seq_len)
    block_mla = TransformerBlock(cfg_mla; use_moe=false)
    @test size(block_mla(randn(Float32, B, T, cfg_mla.dim), freqs_mla[1:T, :])) == (B, T, cfg_mla.dim)
    @test block_mla.attn isa MLAttention

    recurrent = RecurrentBlock(cfg_gqa)
    h = randn(Float32, B, T, cfg_gqa.dim)
    e = randn(Float32, B, T, cfg_gqa.dim)
    out1 = recurrent(copy(h), copy(e), freqs_gqa[1:T, :]; n_loops=1)
    out3 = recurrent(copy(h), copy(e), freqs_gqa[1:T, :]; n_loops=3)
    @test size(out3) == (B, T, cfg_gqa.dim)
    @test !all(isapprox.(out1, out3; atol=1f-6))
end

@testset "OpenMythos GQA" begin
    Random.seed!(5)
    cfg = gqa_cfg()
    model = OpenMythos.OpenMythos(cfg)
    ids = rand(0:(cfg.vocab_size - 1), B, T)

    logits = model(ids)
    @test size(logits) == (B, T, cfg.vocab_size)
    @test !any(isnan, logits)
    @test model.head === model.embed
    @test maximum(get_A(model.recurrent.injection)) < 1.0f0

    shallow = model(ids; n_loops=1)
    deep = model(ids; n_loops=3)
    @test !all(isapprox.(shallow, deep; atol=1f-6))

    cache = Dict{String, Any}()
    logits_no_cache = model(ids; n_loops=2)[:, end, :]
    logits_cached = model(ids; n_loops=2, kv_cache=cache)[:, end, :]
    @test all(isapprox.(logits_no_cache, logits_cached; atol=1f-4))

    single = rand(0:(cfg.vocab_size - 1), B, 1)
    @test size(model(single)) == (B, 1, cfg.vocab_size)

    out = generate(model, ids; max_new_tokens=4, n_loops=2, rng=MersenneTwister(1))
    @test size(out) == (B, T + 4)
end

@testset "OpenMythos MLA" begin
    Random.seed!(6)
    cfg = mla_cfg()
    model = OpenMythos.OpenMythos(cfg)
    ids = rand(0:(cfg.vocab_size - 1), B, T)

    logits = model(ids)
    @test size(logits) == (B, T, cfg.vocab_size)
    @test !any(isnan, logits)
    @test size(generate(model, ids; max_new_tokens=4, n_loops=2, rng=MersenneTwister(2))) == (B, T + 4)
    @test maximum(get_A(model.recurrent.injection)) < 1.0f0

    cache = Dict{String, Any}()
    model(ids; kv_cache=cache)
    mla_entries = [v for v in values(cache) if haskey(v, "c_kv")]
    @test !isempty(mla_entries)
    @test all(size(entry["c_kv"], 3) == cfg.kv_lora_rank for entry in mla_entries)
end

@testset "Attention type swap" begin
    Random.seed!(7)
    cfg_gqa = gqa_cfg()
    cfg_mla = mla_cfg()
    ids = rand(0:(cfg_gqa.vocab_size - 1), B, T)
    logits_gqa = OpenMythos.OpenMythos(cfg_gqa)(ids)
    logits_mla = OpenMythos.OpenMythos(cfg_mla)(ids)
    @test !all(isapprox.(logits_gqa, logits_mla; atol=1f-6))
    @test size(logits_gqa) == (B, T, cfg_gqa.vocab_size)
    @test size(logits_mla) == (B, T, cfg_mla.vocab_size)

    cache_gqa = Dict{String, Any}()
    cache_mla = Dict{String, Any}()
    OpenMythos.OpenMythos(cfg_gqa)(ids[1:1, :]; kv_cache=cache_gqa)
    OpenMythos.OpenMythos(cfg_mla)(ids[1:1, :]; kv_cache=cache_mla)

    cache_bytes(cache) = sum(sizeof(t) for entry in values(cache) for t in values(entry))
    @test cache_bytes(cache_mla) < cache_bytes(cache_gqa)
end
