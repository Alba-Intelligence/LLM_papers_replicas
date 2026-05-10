using Random

@testset "OLMo tiny forward path" begin
    Random.seed!(1)
    cfg = olmo2_tiny()
    model = OLMoModel(cfg; rng=MersenneTwister(1))
    ids = rand(0:(cfg.vocab_size - 1), 2, 8)

    logits = model(ids)
    @test size(logits) == (2, 8, cfg.vocab_size)
    @test !any(isnan, logits)

    cache = Dict{String, Any}()
    cached = model(ids; kv_cache=cache)
    @test size(cached) == size(logits)
    @test !isempty(cache)

    env = chunked_prefill(model, ids; chunk_size=3, envelope=KVCacheEnvelope())
    @test env.start_pos == size(ids, 2) - 1
    @test !isempty(env.cache)

    out = generate(model, ids; max_new_tokens=3, rng=MersenneTwister(2))
    @test size(out) == (2, 11)

    continued = generate(model, ids; max_new_tokens=2, rng=MersenneTwister(3), envelope=env)
    @test size(continued) == (2, 10)
end
