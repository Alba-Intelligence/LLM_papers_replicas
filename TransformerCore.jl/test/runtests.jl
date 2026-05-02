using Test
using TransformerCore

@testset "TransformerCore smoke" begin
    norm = RMSNorm(4)
    x = reshape(Float32.(1:8), 1, 2, 4)
    @test size(norm(x)) == size(x)

    freqs = precompute_rope_freqs(8, 4)
    rope_x = randn(Float32, 1, 4, 2, 8)
    @test size(apply_rope(rope_x, freqs)) == size(rope_x)
end

@testset "TransformerCore training utilities" begin
    sched = WarmupCosineSchedule(2, 6, 1.0f0, 0.1f0)
    @test learning_rate(sched, 0) == 0.0f0
    @test learning_rate(sched, 1) ≈ 0.5f0
    @test learning_rate(sched, 2) ≈ 1.0f0
    @test learning_rate(sched, 6) ≈ 0.1f0

    pairs = chunk_next_token_pairs(collect(0:9), 3)
    @test length(pairs) == 2
    @test pairs[1][1] == [0, 1, 2]
    @test pairs[1][2] == [1, 2, 3]

    text_pairs = text_next_token_pairs(["ab", "cd"], text -> [Int(c) - Int('a') for c in collect(text)], 2)
    @test length(text_pairs) == 1
    @test text_pairs[1][1] == [0, 1]
    @test text_pairs[1][2] == [1, 2]

    batches = batch_next_token_pairs(pairs, 2)
    @test length(batches) == 1
    @test size(batches[1][1]) == (2, 3)
    @test size(batches[1][2]) == (2, 3)

    hidden = reshape(Float32[1, 0, 0, 1], 1, 2, 2)
    head = zeros(Float32, 3, 2)
    targets = reshape(Int[0, 1], 1, :)
    loss, grad = _head_loss_and_grad(hidden, head, targets)
    @test loss ≈ log(3.0f0)
    @test size(grad) == size(head)
end

@testset "KVCacheEnvelope" begin
    env = KVCacheEnvelope()
    @test isempty(env.cache)
    @test env.start_pos == 0
    @test env.capacity_hint == 0
    @test_throws ArgumentError KVCacheEnvelope(Dict{String, Any}(), -1)
    @test_throws ArgumentError KVCacheEnvelope(Dict{String, Any}(), 0, -1)

    hinted = KVCacheEnvelope(; capacity_hint=5)
    reserve_kv_capacity!(hinted, 8)
    reserve_kv_capacity!(hinted, 3)
    @test hinted.capacity_hint == 8

    mktempdir() do dir
        path = joinpath(dir, "cache.jls")
        env.cache["layer0"] = Dict("k" => randn(Float32, 1, 2, 3), "v" => randn(Float32, 1, 2, 3))
        env.start_pos = 7
        env.capacity_hint = 11
        save_kv_cache(env, path)
        @test isfile(path)
        @test !isfile(path * ".tmp")

        loaded = load_kv_cache(path)
        @test loaded.start_pos == env.start_pos
        @test loaded.capacity_hint == env.capacity_hint
        @test keys(loaded.cache) == keys(env.cache)
        @test loaded.cache["layer0"]["k"] == env.cache["layer0"]["k"]
        @test loaded.cache["layer0"]["v"] == env.cache["layer0"]["v"]
    end
end

@testset "AxisAppendBuffer" begin
    chunk1 = reshape(Float32.(1:6), 1, 2, 3)
    chunk2 = reshape(Float32.(7:12), 1, 2, 3)
    buffer = TransformerCore.filled_axis_buffer(chunk1; axis=2, capacity=6)
    @test size(TransformerCore.buffer_view(buffer)) == (1, 2, 3)
    @test TransformerCore.buffer_capacity(buffer) == 6
    @test TransformerCore.buffer_page_count(buffer) == 3

    TransformerCore.append_axis_buffer!(buffer, chunk2)
    @test size(TransformerCore.buffer_view(buffer)) == (1, 4, 3)
    @test Array(TransformerCore.buffer_view(buffer))[:, 1:2, :] == chunk1
    @test Array(TransformerCore.buffer_view(buffer))[:, 3:4, :] == chunk2
    @test TransformerCore.buffer_capacity(buffer) == 6
    @test TransformerCore.buffer_page_count(buffer) == 3
end
