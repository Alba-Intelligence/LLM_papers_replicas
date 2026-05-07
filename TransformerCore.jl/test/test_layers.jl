using Lux
using Random
using Test
using TransformerCore

@testset "Lux-native shared layers" begin
    rng = MersenneTwister(1)

    embed = TokenEmbedding(16, 8)
    ps_embed, st_embed = Lux.setup(rng, embed)
    ids = reshape(Int[0, 1, 2, 3], 2, 2)
    hidden, st_embed_new = Lux.apply(embed, ids, ps_embed, st_embed)
    @test size(hidden) == (2, 2, 8)
    @test st_embed_new == st_embed

    proj = FeatureLinear(8, 5; bias=true)
    ps_proj, st_proj = Lux.setup(rng, proj)
    logits, st_proj_new = Lux.apply(proj, hidden, ps_proj, st_proj)
    @test size(logits) == (2, 2, 5)
    @test st_proj_new == st_proj

    tied = tied_lm_head(hidden, ps_embed.weight)
    @test size(tied) == (2, 2, 16)

    norm = TransformerCore.RMSNorm(8)
    ps_norm, st_norm = Lux.setup(rng, norm)
    normed, st_norm_new = Lux.apply(norm, hidden, ps_norm, st_norm)
    @test size(normed) == size(hidden)
    @test st_norm_new == st_norm
    @test normed ≈ norm(hidden) atol = 1f-6
end
