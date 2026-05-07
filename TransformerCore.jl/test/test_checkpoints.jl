using Lux
using Random
using Test
using TransformerCore

struct CheckpointToyLM{E,P} <: Lux.LuxCore.AbstractLuxLayer
    embed::E
    proj::P
end

function Lux.initialparameters(rng::AbstractRNG, layer::CheckpointToyLM)
    return (
        embed=Lux.initialparameters(rng, layer.embed),
        proj=Lux.initialparameters(rng, layer.proj),
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::CheckpointToyLM)
    return (
        embed=Lux.initialstates(rng, layer.embed),
        proj=Lux.initialstates(rng, layer.proj),
    )
end

function (layer::CheckpointToyLM)(input_ids::AbstractMatrix{<:Integer}, ps, st)
    hidden, st_embed = Lux.apply(layer.embed, input_ids, ps.embed, st.embed)
    logits, st_proj = Lux.apply(layer.proj, hidden, ps.proj, st.proj)
    return logits, (embed=st_embed, proj=st_proj)
end

@testset "Family/mode-aware checkpoints" begin
    layer = CheckpointToyLM(TokenEmbedding(16, 8), FeatureLinear(8, 16; bias=false))
    schedule = WarmupCosineSchedule(0, 3, 0.01f0, 0.01f0)
    state = NextTokenTrainerState(layer; rng=MersenneTwister(24), schedule=schedule, weight_decay=0.0)
    x = reshape(Int[1, 2, 3], 1, :)
    y = reshape(Int[2, 3, 4], 1, :)

    mktempdir() do dir
        train_next_token_step!(state, layer, x, y)
        first_path = save_trainer_checkpoint(state, dir; family="openmythos", mode="full_model", keep_last=2, config=Dict("dim" => 8), metadata=Dict("source" => "test"))
        @test isfile(first_path)
        @test dirname(first_path) == checkpoint_dir(dir; family="openmythos", mode="full_model")
        @test latest_checkpoint(dir; family="openmythos", mode="full_model") == first_path

        train_next_token_step!(state, layer, x, y)
        save_trainer_checkpoint(state, dir; family="openmythos", mode="full_model", keep_last=2)
        train_next_token_step!(state, layer, x, y)
        last_path = save_trainer_checkpoint(state, dir; family="openmythos", mode="full_model", keep_last=2)

        files = filter(name -> endswith(name, ".jls"), readdir(checkpoint_dir(dir; family="openmythos", mode="full_model")))
        @test length(files) == 2

        restored = load_trainer_checkpoint(last_path; expected_family="openmythos", expected_mode="full_model")
        @test restored.state.step == state.step
        @test restored.family == "openmythos"
        @test restored.mode == "full_model"
        @test restored.state.schedule == state.schedule
    end
end
