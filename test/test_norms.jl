using Test
using Statistics

@testset "RMSNorm" begin
    norm = RMSNorm(64)
    x = randn(Float32, 2, 8, 64)
    out = norm(x)
    @test size(out) == size(x)

    norm_ones = RMSNorm(64)
    x2 = randn(Float32, 4, 64)
    out2 = norm_ones(x2)
    rms = sqrt.(mean(abs2, out2; dims=2))
    @test all(isapprox.(vec(rms), ones(Float32, size(rms, 1)); atol=1f-4))

    @test length(norm.weight) == 64
end
