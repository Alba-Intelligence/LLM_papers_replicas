using Random

@testset "DeepSeekV4 tiny forward path" begin
    Random.seed!(1)
    cfg = deepseek_v4_tiny()
    model = DeepSeekV4Model(cfg)
    ids = rand(0:(cfg.vocab_size - 1), 2, 8)

    logits = model(ids)
    @test size(logits) == (2, 8, cfg.vocab_size)
    @test !any(isnan, logits)

    mtp = mtp_logits(model, ids)
    @test size(mtp) == (2, 8, cfg.mtp_tokens, cfg.vocab_size)

    cache = Dict{String, Any}()
    cached = model(ids; kv_cache=cache)
    @test size(cached) == size(logits)
    @test !isempty(cache)

    out = generate(model, ids; max_new_tokens=3, rng=MersenneTwister(2))
    @test size(out) == (2, 11)
end
