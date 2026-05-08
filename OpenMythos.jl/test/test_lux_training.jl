using Random
using Test

@testset "LuxFullModelTrainerState reduces repeated-batch loss" begin
    Random.seed!(71)
    cfg = bootstrap_full_model_training_config(32; seq_len=4, attn_type="gqa")
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(71))
    schedule = WarmupCosineSchedule(0, 4, 0.01f0, 0.01f0)
    state = LuxFullModelTrainerState(model; schedule=schedule, weight_decay=0.0, n_loops=cfg.max_loop_iters)

    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    initial_loss = lux_full_model_loss(state, x, y)
    embed_before = copy(state.trainer.ps.embed.weight)
    for _ in 1:4
        train_lux_full_model_step!(state, x, y)
    end
    final_loss = lux_full_model_loss(state, x, y)

    @test final_loss < initial_loss
    @test state.trainer.step == 4
    @test !all(isapprox.(state.trainer.ps.embed.weight, embed_before; atol=1f-6))
    @test size(lux_full_model_logits(state, x)) == (1, 4, cfg.vocab_size)
end

@testset "LuxFullModelTrainerState supports MLA and shared checkpoints" begin
    Random.seed!(72)
    cfg = bootstrap_full_model_training_config(32; seq_len=4, attn_type="mla", n_shared_experts=1)
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(72))
    schedule = WarmupCosineSchedule(0, 2, 0.01f0, 0.01f0)
    state = LuxFullModelTrainerState(model; schedule=schedule, weight_decay=0.0, n_loops=cfg.max_loop_iters)
    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    initial_loss = lux_full_model_loss(state, x, y)
    train_lux_full_model_step!(state, x, y)
    train_lux_full_model_step!(state, x, y)
    final_loss = lux_full_model_loss(state, x, y)
    @test final_loss < initial_loss

    mktempdir() do dir
        batches = [(x, y)]
        io = IOBuffer()
        state2 = LuxFullModelTrainerState(model; schedule=schedule, weight_decay=0.0, n_loops=cfg.max_loop_iters)
        metrics = train_lux_full_model!(state2, batches; total_steps=2, log_every=1, ckpt_root=dir, ckpt_every=2, io=io)
        @test metrics.step == 2
        latest = latest_checkpoint(dir; family="openmythos", mode="full_model_lux")
        @test latest !== nothing
        restored = TransformerCore.load_trainer_checkpoint(latest; expected_family="openmythos", expected_mode="full_model_lux")
        @test restored.state.step == 2
        @test restored.config == cfg

        manual_path = save_lux_full_model_checkpoint(state2, dir; keep_last=2, metadata=Dict("source" => "manual-test"))
        @test isfile(manual_path)
        loaded = load_lux_full_model_checkpoint(manual_path)
        @test loaded.trainer.step == state2.trainer.step
        @test loaded.layer.model.cfg == state2.layer.model.cfg
        @test loaded.layer.n_loops == state2.layer.n_loops
    end
end
