using Test
using Random

@testset "loop_index_embedding" begin
    h = randn(Float32, 2, 8, 64)
    out = loop_index_embedding(h, 0, 8)
    @test size(out) == size(h)

    zeros_h = zeros(Float32, 1, 1, 64)
    out0 = loop_index_embedding(zeros_h, 0, 8)
    out1 = loop_index_embedding(zeros_h, 1, 8)
    @test !all(isapprox.(out0, out1; atol=1f-6))
    @test all(out1[:, :, 9:end] .== 0)
end

@testset "LoRAAdapter" begin
    Random.seed!(11)
    lora = LoRAAdapter(64, 8, 10)
    x = randn(Float32, 2, 8, 64)
    out = lora(x, 0)
    @test size(out) == size(x)

    out0 = lora(x, 0)
    out1 = lora(x, 1)
    @test !all(isapprox.(out0, out1; atol=1f-6))

    out_clamped = lora(x, 999)
    out_last = lora(x, 9)
    @test all(isapprox.(out_clamped, out_last; atol=1f-6))
end

@testset "LTIInjection" begin
    inj = LTIInjection(64)
    h = randn(Float32, 2, 8, 64)
    e = randn(Float32, 2, 8, 64)
    t = randn(Float32, 2, 8, 64)
    out = inj(h, e, t)
    @test size(out) == size(h)

    A = get_A(inj)
    @test maximum(A) < 1.0f0
    @test minimum(A) > 0.0f0

    inj_big = LTIInjection(64)
    inj_big.log_A .= 1.0f6
    inj_big.log_dt .= 1.0f6
    A_big = get_A(inj_big)
    @test maximum(A_big) < 1.0f0
    @test minimum(A_big) > 0.0f0
end

@testset "ACTHalting" begin
    Random.seed!(13)
    act = ACTHalting(64)
    h = randn(Float32, 2, 8, 64)
    p = act(h)
    @test size(p) == (2, 8)
    @test minimum(p) >= 0.0f0
    @test maximum(p) <= 1.0f0
end
