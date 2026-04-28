using Test

@testset "Variant presets" begin
    @test mythos_1b().dim == 2_048
    @test mythos_3b().dim == 3_072
    @test mythos_10b().max_loop_iters == 24
    @test mythos_50b().n_experts == 256
    @test mythos_100b().max_output_tokens == 131_072
    @test mythos_500b().vocab_size == 100_000
    @test mythos_1t().rope_theta == 2_000_000.0f0
end
