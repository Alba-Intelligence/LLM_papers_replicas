using Lux
using Random

@testset "WarmupCosineSchedule" begin
    sched = WarmupCosineSchedule(2, 6, 1.0f0, 0.1f0)
    @test learning_rate(sched, 0) == 0.0f0
    @test learning_rate(sched, 1) ≈ 0.5f0
    @test learning_rate(sched, 2) ≈ 1.0f0
    @test learning_rate(sched, 6) ≈ 0.1f0
end

@testset "Token chunking and batching" begin
    pairs = chunk_next_token_pairs(collect(0:9), 3)
    @test length(pairs) == 2
    @test pairs[1][1] == [0, 1, 2]
    @test pairs[1][2] == [1, 2, 3]
    @test pairs[2][1] == [4, 5, 6]
    @test pairs[2][2] == [5, 6, 7]

    text_pairs = text_next_token_pairs(["ab", "cd"], text -> [Int(c) - Int('a') for c in collect(text)], 2)
    @test length(text_pairs) == 1
    @test text_pairs[1][1] == [0, 1]
    @test text_pairs[1][2] == [1, 2]

    batches = batch_next_token_pairs(pairs, 2)
    @test length(batches) == 1
    @test size(batches[1][1]) == (2, 3)
    @test size(batches[1][2]) == (2, 3)
end

@testset "DeepSeek head-only trainer step reduces repeated-batch loss" begin
    Random.seed!(21)
    cfg = bootstrap_deepseek_training_config(64; seq_len=4)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(21))
    schedule = WarmupCosineSchedule(0, 8, 0.05f0, 0.05f0)
    state = DeepSeekHeadTrainerState(model; schedule=schedule, weight_decay=0.0)

    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    initial_loss = deepseek_head_loss(state, x, y)
    for _ in 1:8
        train_deepseek_head_only_step!(state, x, y)
    end
    final_loss = deepseek_head_loss(state, x, y)

    @test final_loss < initial_loss
    @test state.step == 8
    @test !all(isapprox.(state.head, model.head; atol=1f-6))

    lux_layer = LuxHeadOnlyDeepSeekV4(model)
    logits_from_lux, _ = Lux.apply(lux_layer, x, (head=state.head,), Lux.initialstates(MersenneTwister(21), lux_layer))
    @test logits_from_lux == deepseek_head_logits(state, x)
end

@testset "DeepSeek checkpoint roundtrip and pruning" begin
    Random.seed!(22)
    cfg = bootstrap_deepseek_training_config(48; seq_len=4)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(22))
    schedule = WarmupCosineSchedule(0, 4, 0.01f0, 0.01f0)
    state = DeepSeekHeadTrainerState(model; schedule=schedule, weight_decay=0.0)
    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    mktempdir() do dir
        train_deepseek_head_only_step!(state, x, y)
        train_deepseek_head_only_step!(state, x, y)
        first_path = save_deepseek_checkpoint(state, dir; keep_last=2, metadata=Dict("source" => "test"))
        @test isfile(first_path)

        train_deepseek_head_only_step!(state, x, y)
        save_deepseek_checkpoint(state, dir; keep_last=2)
        train_deepseek_head_only_step!(state, x, y)
        last_path = save_deepseek_checkpoint(state, dir; keep_last=2)

        files = filter(name -> endswith(name, ".jls"), readdir(dir))
        @test length(files) == 2
        @test latest_checkpoint(dir) == last_path

        restored = load_deepseek_checkpoint(last_path, model)
        @test restored.step == state.step
        @test restored.head == state.head
        @test typeof(restored.opt_state) == typeof(state.opt_state)
    end
end

@testset "DeepSeek training loop saves final checkpoint" begin
    Random.seed!(23)
    cfg = bootstrap_deepseek_training_config(48; seq_len=4)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(23))
    schedule = WarmupCosineSchedule(0, 3, 0.02f0, 0.02f0)
    state = DeepSeekHeadTrainerState(model; schedule=schedule, weight_decay=0.0)
    pairs = chunk_next_token_pairs(collect(0:14), 4)
    batches = batch_next_token_pairs(pairs, 2)

    mktempdir() do dir
        io = IOBuffer()
        metrics = train_deepseek_head_only!(state, batches; total_steps=3, log_every=1, ckpt_dir=dir, ckpt_every=2, io=io)
        @test metrics.step == 3
        @test occursin("step 3/3", String(take!(io)))
        latest = latest_checkpoint(dir)
        @test latest !== nothing
        @test endswith(latest, "step_0000003.jls")
    end
end
