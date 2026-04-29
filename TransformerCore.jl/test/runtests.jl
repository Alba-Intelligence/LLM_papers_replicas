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
