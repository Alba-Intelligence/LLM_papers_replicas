using Random

@testset "Engram token lookup compression" begin
    lookup = build_engram_token_lookup([" Foo", "foo", "bar", "bar  ", "BAZ", "baz"])
    @test lookup[1] == lookup[2]
    @test lookup[3] == lookup[4]
    @test lookup[5] == lookup[6]
    @test maximum(lookup) == 2
end

@testset "DeepSeekV4 Engram forward path" begin
    Random.seed!(2)
    cfg = deepseek_v4_tiny_engram()
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(2))
    ids = rand(0:(cfg.vocab_size - 1), 2, 8)

    logits = model(ids)
    @test size(logits) == (2, 8, cfg.vocab_size)
    @test !any(isnan, logits)
    @test model.blocks[1].engram !== nothing
    @test model.blocks[2].engram === nothing
    @test model.blocks[3].engram !== nothing
end

@testset "DeepSeekV4 Engram compressed lookup path" begin
    Random.seed!(3)
    vocab_tokens = [" Foo", "foo", "Bar", "bar", "Baz", "baz", "Qux", "qux"]
    lookup = build_engram_token_lookup(vocab_tokens)
    cfg = bootstrap_deepseek_training_config(length(vocab_tokens); seq_len=4, with_engram=true, engram_token_lookup=lookup)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(3))
    ids = reshape([0, 1, 2, 3], 1, :)

    logits = model(ids)
    @test size(logits) == (1, 4, cfg.vocab_size)
    @test !any(isnan, logits)
    @test model.blocks[1].engram !== nothing
end
