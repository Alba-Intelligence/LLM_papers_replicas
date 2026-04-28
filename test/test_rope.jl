using Test
using LinearAlgebra
using Random

@testset "RoPE" begin
    freqs = precompute_rope_freqs(16, 32)
    @test size(freqs) == (32, 8)
    @test all(isapprox.(abs.(freqs), ones(Float32, size(freqs)); atol=1f-6))
    @test all(isapprox.(freqs[1, :], ones(ComplexF32, 8); atol=1f-6))

    x = randn(Float32, 2, 8, 4, 16)
    out = apply_rope(x, freqs[1:8, :])
    @test size(out) == size(x)

    xnorm = sqrt.(sum(abs2, x; dims=4))
    onorm = sqrt.(sum(abs2, out; dims=4))
    @test all(isapprox.(xnorm, onorm; atol=1f-5))

    xones = ones(Float32, 1, 2, 1, 16)
    outones = apply_rope(xones, freqs[1:2, :])
    @test !all(isapprox.(outones[:, 1:1, :, :], outones[:, 2:2, :, :]; atol=1f-6))

    Random.seed!(7)
    xa = randn(Float32, 1, 4, 2, 16)
    xb = randn(Float32, 1, 4, 2, 16)
    solo = apply_rope(xa, freqs[1:4, :])
    batched = apply_rope(cat(xa, xb; dims=1), freqs[1:4, :])[1:1, :, :, :]
    @test all(isapprox.(solo, batched; atol=1f-6))

    xhead = repeat(randn(Float32, 1, 4, 1, 16), 1, 1, 3, 1)
    outhead = apply_rope(xhead, freqs[1:4, :])
    @test all(isapprox.(outhead[:, :, 1:1, :], outhead[:, :, 2:2, :]; atol=1f-6))
    @test all(isapprox.(outhead[:, :, 2:2, :], outhead[:, :, 3:3, :]; atol=1f-6))

    dim = 16
    max_len = 32
    freqs_big = precompute_rope_freqs(dim, max_len)
    Random.seed!(42)
    q = randn(Float32, 1, 1, 1, dim)
    k = randn(Float32, 1, 1, 1, dim)

    function rope_at(tensor, pos)
        seq = zeros(Float32, 1, pos + 1, 1, dim)
        seq[1, pos + 1, 1, :] .= tensor[1, 1, 1, :]
        return apply_rope(seq, freqs_big[1:(pos + 1), :])[:, (pos + 1):(pos + 1), :, :]
    end

    dot_3_9 = sum(rope_at(q, 3) .* rope_at(k, 9))
    dot_1_7 = sum(rope_at(q, 1) .* rope_at(k, 7))
    @test isapprox(dot_3_9, dot_1_7; atol=1f-5)

    freqs_short = precompute_rope_freqs(8, 4)
    xshort = randn(Float32, 1, 8, 2, 8)
    @test_throws DimensionMismatch apply_rope(xshort, freqs_short)
end
