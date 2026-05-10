using Random

@testset "OLMo full-model trainer step reduces repeated-batch loss" begin
    Random.seed!(21)
    cfg = bootstrap_olmo_training_config(64; seq_len=4)
    model = OLMoModel(cfg; rng=MersenneTwister(21))
    schedule = WarmupCosineSchedule(0, 4, 0.01f0, 0.01f0)
    state = OLMoFullModelTrainerState(model; schedule=schedule, weight_decay=0.0)

    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)
    head_before = copy(model.head)

    initial_loss = olmo_loss(state, x, y)
    train_olmo_step!(state, x, y)
    final_loss = olmo_loss(state, x, y)

    @test final_loss < initial_loss
    @test state.step == 1
    @test size(olmo_logits(state, x)) == (1, 4, cfg.vocab_size)
    @test !all(isapprox.(state.model.head, head_before; atol=1f-6))
end

@testset "OLMo checkpoint roundtrip and training loop" begin
    Random.seed!(22)
    cfg = bootstrap_olmo_training_config(48; seq_len=4)
    model = OLMoModel(cfg; rng=MersenneTwister(22))
    schedule = WarmupCosineSchedule(0, 3, 0.02f0, 0.02f0)
    state = OLMoFullModelTrainerState(model; schedule=schedule, weight_decay=0.0)
    pairs = chunk_next_token_pairs(collect(0:14), 4)
    batches = batch_next_token_pairs(pairs, 2)

    mktempdir() do dir
        train_olmo_step!(state, batches[1][1], batches[1][2])
        path = save_olmo_checkpoint(state, dir; keep_last=2, metadata=Dict("source" => "test"))
        @test isfile(path)
        @test occursin(joinpath("olmo", "full_model"), path)

        restored = load_olmo_checkpoint(path)
        @test restored.step == state.step
        @test all(field -> getfield(restored.model.cfg, field) == getfield(state.model.cfg, field), fieldnames(OLMoConfig))
        @test olmo_logits(restored, batches[1][1]) == olmo_logits(state, batches[1][1])

        io = IOBuffer()
        metrics = train_olmo!(restored, batches; total_steps=3, log_every=1, ckpt_root=dir, ckpt_every=2, io=io)
        @test metrics.step == 3
        @test occursin("step 3/3", String(take!(io)))
        latest = latest_checkpoint(dir; family="olmo", mode="full_model")
        @test latest !== nothing
        @test endswith(latest, "step_0000003.jls")
    end
end
