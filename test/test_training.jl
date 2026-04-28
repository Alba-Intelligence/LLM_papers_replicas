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

@testset "Head-only trainer step reduces repeated-batch loss" begin
    Random.seed!(11)
    cfg = bootstrap_training_config(64; seq_len=6, attn_type="gqa")
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(11))
    schedule = WarmupCosineSchedule(0, 12, 0.05f0, 0.05f0)
    state = HeadOnlyTrainerState(model; schedule=schedule, weight_decay=0.0)

    x = [
        1 2 3 4 5 6
        2 3 4 5 6 7
    ]
    y = [
        2 3 4 5 6 7
        3 4 5 6 7 8
    ]

    initial_loss = head_only_loss(state, x, y; n_loops=2)
    for _ in 1:12
        train_head_only_step!(state, x, y; n_loops=2)
    end
    final_loss = head_only_loss(state, x, y; n_loops=2)

    @test final_loss < initial_loss
    @test state.step == 12
    @test !all(isapprox.(state.head, model.head; atol=1f-6))
end

@testset "Checkpoint roundtrip and pruning" begin
    Random.seed!(12)
    cfg = bootstrap_training_config(32; seq_len=4, attn_type="gqa")
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(12))
    schedule = WarmupCosineSchedule(0, 4, 0.01f0, 0.01f0)
    state = HeadOnlyTrainerState(model; schedule=schedule, weight_decay=0.0)
    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    mktempdir() do dir
        train_head_only_step!(state, x, y; n_loops=1)
        train_head_only_step!(state, x, y; n_loops=1)
        first_path = save_head_only_checkpoint(state, dir; keep_last=2, metadata=Dict("source" => "test"))
        @test isfile(first_path)

        train_head_only_step!(state, x, y; n_loops=1)
        save_head_only_checkpoint(state, dir; keep_last=2)
        train_head_only_step!(state, x, y; n_loops=1)
        last_path = save_head_only_checkpoint(state, dir; keep_last=2)

        files = filter(name -> endswith(name, ".jls"), readdir(dir))
        @test length(files) == 2
        @test latest_checkpoint(dir) == last_path

        restored = load_head_only_checkpoint(last_path, model)
        @test restored.step == state.step
        @test restored.head == state.head
        @test restored.adam_m == state.adam_m
        @test restored.adam_v == state.adam_v
    end
end

@testset "Training loop saves final checkpoint" begin
    Random.seed!(13)
    cfg = bootstrap_training_config(48; seq_len=4, attn_type="gqa")
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(13))
    schedule = WarmupCosineSchedule(0, 3, 0.02f0, 0.02f0)
    state = HeadOnlyTrainerState(model; schedule=schedule, weight_decay=0.0)
    pairs = chunk_next_token_pairs(collect(0:14), 4)
    batches = batch_next_token_pairs(pairs, 2)

    mktempdir() do dir
        io = IOBuffer()
        metrics = train_head_only!(state, batches; total_steps=3, n_loops=1, log_every=1, ckpt_dir=dir, ckpt_every=2, io=io)
        @test metrics.step == 3
        @test occursin("step 3/3", String(take!(io)))
        latest = latest_checkpoint(dir)
        @test latest !== nothing
        @test endswith(latest, "step_0000003.jls")
    end
end
