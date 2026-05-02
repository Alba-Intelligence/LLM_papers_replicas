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

    lux_layer = LuxHeadOnlyOpenMythos(model; n_loops=2)
    logits_from_lux, _ = Lux.apply(lux_layer, x, (head=state.head,), Lux.initialstates(MersenneTwister(11), lux_layer))
    @test logits_from_lux == head_only_logits(state, x; n_loops=2)
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
        @test typeof(restored.opt_state) == typeof(state.opt_state)
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

@testset "Full-model trainer step reduces repeated-batch loss" begin
    Random.seed!(14)
    cfg = bootstrap_full_model_training_config(32; seq_len=4)
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(14))
    schedule = WarmupCosineSchedule(0, 4, 0.01f0, 0.01f0)
    state = FullModelTrainerState(model; schedule=schedule, weight_decay=0.0, n_loops=cfg.max_loop_iters)

    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    initial_loss = full_model_loss(state, x, y)
    original_head = copy(state.model.head)
    for _ in 1:4
        train_full_model_step!(state, x, y)
    end
    final_loss = full_model_loss(state, x, y)

    @test final_loss < initial_loss
    @test state.step == 4
    @test !all(isapprox.(state.model.head, original_head; atol=1f-6))
end

@testset "Full-model trainer supports MLA and shared experts" begin
    Random.seed!(17)
    cfg = bootstrap_full_model_training_config(32; seq_len=4, attn_type="mla", n_shared_experts=1)
    @test cfg.attn_type == "mla"
    @test cfg.n_experts == 1
    @test cfg.n_shared_experts == 1
    @test cfg.n_experts_per_tok == 1

    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(17))
    schedule = WarmupCosineSchedule(0, 4, 0.01f0, 0.01f0)
    state = FullModelTrainerState(model; schedule=schedule, weight_decay=0.0, n_loops=cfg.max_loop_iters)

    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    initial_loss = full_model_loss(state, x, y)
    for _ in 1:4
        train_full_model_step!(state, x, y)
    end
    final_loss = full_model_loss(state, x, y)

    @test final_loss < initial_loss
    @test state.model.cfg.attn_type == "mla"
    @test state.model.cfg.n_shared_experts == 1

    mktempdir() do dir
        path = save_full_model_checkpoint(state, dir; keep_last=1)
        restored = load_full_model_checkpoint(path)
        @test restored.model.cfg.attn_type == "mla"
        @test restored.model.cfg.n_shared_experts == 1
        @test restored.model.head == state.model.head
    end
end

@testset "Full-model checkpoint roundtrip and pruning" begin
    Random.seed!(15)
    cfg = bootstrap_full_model_training_config(24; seq_len=4)
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(15))
    schedule = WarmupCosineSchedule(0, 3, 0.005f0, 0.005f0)
    state = FullModelTrainerState(model; schedule=schedule, weight_decay=0.0, n_loops=cfg.max_loop_iters)
    x = reshape([1, 2, 3, 4], 1, :)
    y = reshape([2, 3, 4, 5], 1, :)

    mktempdir() do dir
        train_full_model_step!(state, x, y)
        first_path = save_full_model_checkpoint(state, dir; keep_last=2, metadata=Dict("source" => "test"))
        @test isfile(first_path)

        train_full_model_step!(state, x, y)
        save_full_model_checkpoint(state, dir; keep_last=2)
        train_full_model_step!(state, x, y)
        last_path = save_full_model_checkpoint(state, dir; keep_last=2)

        files = filter(name -> endswith(name, ".jls"), readdir(dir))
        @test length(files) == 2
        @test latest_checkpoint(dir) == last_path

        restored = load_full_model_checkpoint(last_path)
        @test restored.step == state.step
        @test restored.n_loops == state.n_loops
        @test restored.model.head == state.model.head
    end
end

@testset "Full-model training loop saves final checkpoint" begin
    Random.seed!(16)
    cfg = bootstrap_full_model_training_config(24; seq_len=4)
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(16))
    schedule = WarmupCosineSchedule(0, 2, 0.005f0, 0.005f0)
    state = FullModelTrainerState(model; schedule=schedule, weight_decay=0.0, n_loops=cfg.max_loop_iters)
    pairs = chunk_next_token_pairs(collect(0:10), 4)
    batches = batch_next_token_pairs(pairs, 1)

    mktempdir() do dir
        io = IOBuffer()
        metrics = train_full_model!(state, batches; total_steps=2, n_loops=cfg.max_loop_iters, log_every=1, ckpt_dir=dir, ckpt_every=2, io=io)
        @test metrics.step == 2
        @test occursin("step 2/2", String(take!(io)))
        latest = latest_checkpoint(dir)
        @test latest !== nothing
        @test endswith(latest, "step_0000002.jls")
    end
end
