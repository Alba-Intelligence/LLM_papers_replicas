using Test

@testset "MythosConfig" begin
    cfg = MythosConfig()
    @test cfg.attn_type == "mla"
    @test cfg.dim == 2048
    @test cfg.max_loop_iters == 16
    @test cfg.rope_theta == 500_000.0f0
end
