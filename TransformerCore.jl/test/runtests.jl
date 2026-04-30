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
