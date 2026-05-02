using Random

cache_capacities(cache) = [OpenMythos.TransformerCore.buffer_capacity(buf) for entry in values(cache) for buf in values(entry)]
cache_pages(cache) = [OpenMythos.TransformerCore.buffer_page_count(buf) for entry in values(cache) for buf in values(entry)]

@testset "OpenMythos KV cache envelope" begin
    Random.seed!(31)
    cfg = bootstrap_training_config(64; seq_len=16, attn_type="gqa")
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(31))
    ids = rand(0:(cfg.vocab_size - 1), 1, 8)

    baseline = generate(model, ids; max_new_tokens=4, n_loops=2, rng=MersenneTwister(1))

    env = chunked_prefill(model, ids; chunk_size=3, n_loops=2, envelope=KVCacheEnvelope(; capacity_hint=size(ids, 2) + 4))
    @test env.start_pos == size(ids, 2) - 1
    @test !isempty(cache_capacities(env.cache))
    @test all(>=(env.capacity_hint), cache_capacities(env.cache))
    @test any(>(1), cache_pages(env.cache))
    with_prefill = generate(model, ids; max_new_tokens=4, n_loops=2, rng=MersenneTwister(1), envelope=env)
    @test with_prefill == baseline
    @test env.start_pos == size(with_prefill, 2) - 1
    @test all(>=(env.capacity_hint), cache_capacities(env.cache))

    mktempdir() do dir
        path = joinpath(dir, "openmythos_cache.jls")
        save_kv_cache(chunked_prefill(model, ids; chunk_size=2, n_loops=2), path)
        loaded = load_kv_cache(path)
        resumed = generate(model, ids; max_new_tokens=4, n_loops=2, rng=MersenneTwister(2), envelope=loaded)
        fresh = generate(model, ids; max_new_tokens=4, n_loops=2, rng=MersenneTwister(2), envelope=chunked_prefill(model, ids; chunk_size=2, n_loops=2))
        @test resumed == fresh
    end

    single = rand(0:(cfg.vocab_size - 1), 1, 1)
    single_env = chunked_prefill(model, single; chunk_size=2, n_loops=2)
    @test single_env.start_pos == 0
    @test size(generate(model, single; max_new_tokens=2, n_loops=2, rng=MersenneTwister(3), envelope=single_env)) == (1, 3)

    too_long = rand(0:(cfg.vocab_size - 1), 1, cfg.max_seq_len + 2)
    @test_throws BoundsError begin
        chunked_prefill(model, too_long; chunk_size=4, n_loops=2)
    end
end
