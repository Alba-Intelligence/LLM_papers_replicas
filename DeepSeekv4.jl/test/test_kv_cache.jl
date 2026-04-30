using Random

@testset "DeepSeekV4 KV cache envelope" begin
    Random.seed!(41)
    cfg = bootstrap_deepseek_training_config(64; seq_len=16)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(41))
    ids = rand(0:(cfg.vocab_size - 1), 1, 8)

    baseline = generate(model, ids; max_new_tokens=4, rng=MersenneTwister(4))

    env = chunked_prefill(model, ids; chunk_size=3)
    @test env.start_pos == size(ids, 2) - 1
    with_prefill = generate(model, ids; max_new_tokens=4, rng=MersenneTwister(4), envelope=env)
    @test with_prefill == baseline
    @test env.start_pos == size(with_prefill, 2) - 1

    mktempdir() do dir
        path = joinpath(dir, "deepseek_cache.jls")
        save_kv_cache(chunked_prefill(model, ids; chunk_size=2), path)
        loaded = load_kv_cache(path)
        resumed = generate(model, ids; max_new_tokens=4, rng=MersenneTwister(5), envelope=loaded)
        fresh = generate(model, ids; max_new_tokens=4, rng=MersenneTwister(5), envelope=chunked_prefill(model, ids; chunk_size=2))
        @test resumed == fresh
    end

    single = rand(0:(cfg.vocab_size - 1), 1, 1)
    single_env = chunked_prefill(model, single; chunk_size=2)
    @test single_env.start_pos == 0
    @test size(generate(model, single; max_new_tokens=2, rng=MersenneTwister(6), envelope=single_env)) == (1, 3)

    too_long = rand(0:(cfg.vocab_size - 1), 1, cfg.max_seq_len + 2)
    @test_throws BoundsError begin
        chunked_prefill(model, too_long; chunk_size=4)
    end
end
