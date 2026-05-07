using Lux
using Random
using Test
using TransformerCore

struct TinyLM{E,P} <: Lux.LuxCore.AbstractLuxLayer
    embed::E
    proj::P
end

function Lux.initialparameters(rng::AbstractRNG, layer::TinyLM)
    return (
        embed=Lux.initialparameters(rng, layer.embed),
        proj=Lux.initialparameters(rng, layer.proj),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::TinyLM)
    return (
        embed=Lux.initialstates(rng, layer.embed),
        proj=Lux.initialstates(rng, layer.proj),
    )
end

function (layer::TinyLM)(input_ids::AbstractMatrix{<:Integer}, ps, st)
    hidden, st_embed = Lux.apply(layer.embed, input_ids, ps.embed, st.embed)
    logits, st_proj = Lux.apply(layer.proj, hidden, ps.proj, st.proj)
    return logits, (embed=st_embed, proj=st_proj)
end

@testset "NextTokenTrainerState and training loop" begin
    Random.seed!(22)
    layer = TinyLM(TokenEmbedding(32, 8), FeatureLinear(8, 32; bias=false))
    schedule = WarmupCosineSchedule(0, 12, 0.05f0, 0.05f0)
    state = NextTokenTrainerState(layer; rng=MersenneTwister(22), schedule=schedule, weight_decay=0.0)

    x = [
        1 2 3 4
        2 3 4 5
    ]
    y = [
        2 3 4 5
        3 4 5 6
    ]

    initial_loss = next_token_loss(layer, state, x, y)
    for _ in 1:12
        train_next_token_step!(state, layer, x, y)
    end
    final_loss = next_token_loss(layer, state, x, y)

    @test final_loss < initial_loss
    @test state.step == 12
    @test size(next_token_logits(layer, state, x)) == (2, 4, 32)
end

@testset "Gradient masking supports head-only-style freezing" begin
    Random.seed!(23)
    layer = TinyLM(TokenEmbedding(32, 8), FeatureLinear(8, 32; bias=false))
    schedule = WarmupCosineSchedule(0, 4, 0.05f0, 0.05f0)
    state = NextTokenTrainerState(layer; rng=MersenneTwister(23), schedule=schedule, weight_decay=0.0)

    mask = parameter_mask(state.ps, (path, _) -> path == (:proj, :weight))

    embed_before = copy(state.ps.embed.weight)
    proj_before = copy(state.ps.proj.weight)

    x = reshape(Int[1, 2, 3, 4], 1, :)
    y = reshape(Int[2, 3, 4, 5], 1, :)
    train_next_token_step!(state, layer, x, y; grad_mask=mask)

    @test state.ps.embed.weight == embed_before
    @test !all(isapprox.(state.ps.proj.weight, proj_before; atol=1f-6))
end
