using Random
using DeepSeekV4

const TRAIN_TEXT_FILE = get(ENV, "DEEPSEEK_V4_TRAIN_TEXT_FILE", "")
const TRAIN_TEXT = get(ENV, "DEEPSEEK_V4_TRAIN_TEXT", "")
const TRAIN_MODE = get(ENV, "DEEPSEEK_V4_TRAIN_MODE", "head_only")
const USE_ENGRAM = get(ENV, "DEEPSEEK_V4_TRAIN_USE_ENGRAM", "0") == "1"
const CKPT_DIR = get(ENV, "DEEPSEEK_V4_TRAIN_CKPT_DIR", "")

const VOCAB_SIZE = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_VOCAB_SIZE", "512"))
const SEQ_LEN = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_SEQ_LEN", "32"))
const BATCH_SIZE = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_BATCH_SIZE", "2"))
const TOTAL_STEPS = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_TOTAL_STEPS", "8"))
const WARMUP_STEPS = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_WARMUP_STEPS", "2"))
const LR = parse(Float32, get(ENV, "DEEPSEEK_V4_TRAIN_LR", "0.02"))
const WEIGHT_DECAY = parse(Float32, get(ENV, "DEEPSEEK_V4_TRAIN_WEIGHT_DECAY", "0.0"))
const CKPT_EVERY = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_CKPT_EVERY", "4"))
const KEEP_LAST = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_KEEP_LAST", "3"))
const RNG_SEED = parse(Int, get(ENV, "DEEPSEEK_V4_TRAIN_SEED", "1"))

function _default_texts()
    return [
        "DeepSeek V4 in Julia now has both head only and tiny full model bootstrap training paths.",
        "The optional Engram branch hashes local token n-grams into a gated residual memory path.",
        "The current full model bootstrap path optimizes the main LM logits end to end on tiny configs.",
        "Shared schedules batching and checkpoint helpers now live in TransformerCore for reuse across packages.",
        "This script is a smoke trainer for local text batches encoded through a simple byte fallback.",
    ]
end

function _load_local_texts()
    if !isempty(TRAIN_TEXT)
        return [TRAIN_TEXT]
    end
    if !isempty(TRAIN_TEXT_FILE)
        return [read(TRAIN_TEXT_FILE, String)]
    end
    return _default_texts()
end

_encode_bytes(text::AbstractString, vocab_size::Integer) = Int[mod(Int(b), vocab_size) for b in codeunits(text)]

function _load_batches(vocab_size::Integer)
    texts = _load_local_texts()
    println("Building local byte-encoded batches...")
    pairs = text_next_token_pairs(texts, text -> _encode_bytes(text, vocab_size), SEQ_LEN)
    return batch_next_token_pairs(pairs, BATCH_SIZE; drop_last=false)
end

function _checkpoint_dir(mode::AbstractString)
    return isempty(CKPT_DIR) ? joinpath("checkpoints", mode) : CKPT_DIR
end

function main()
    TRAIN_MODE in ("head_only", "full_model") || error("DEEPSEEK_V4_TRAIN_MODE must be head_only or full_model")
    cfg = TRAIN_MODE == "full_model" ?
        bootstrap_deepseek_full_model_training_config(VOCAB_SIZE; seq_len=SEQ_LEN, with_engram=USE_ENGRAM) :
        bootstrap_deepseek_training_config(VOCAB_SIZE; seq_len=SEQ_LEN, with_engram=USE_ENGRAM)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(RNG_SEED))
    schedule = WarmupCosineSchedule(WARMUP_STEPS, TOTAL_STEPS, LR, LR * 0.1f0)
    ckpt_dir = _checkpoint_dir(TRAIN_MODE)

    batches = _load_batches(cfg.vocab_size)
    isempty(batches) && error("no training batches available; provide longer local text input")

    latest = latest_checkpoint(ckpt_dir)

    println("DeepSeek vocab size: $(cfg.vocab_size) | seq_len: $(SEQ_LEN) | batch_size: $(BATCH_SIZE) | total_steps: $(TOTAL_STEPS)")
    println("Training mode: $(TRAIN_MODE == \"full_model\" ? \"tiny full-model bootstrap\" : \"Lux-backed head-only bootstrap\")")
    println("Engram branch: $(USE_ENGRAM ? \"enabled\" : \"disabled\")")
    latest !== nothing && println("Resuming from $(latest)")

    metrics = if TRAIN_MODE == "full_model"
        state = latest === nothing ? DeepSeekFullModelTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY) : load_deepseek_full_model_checkpoint(latest)
        train_deepseek_full_model!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            log_every=1,
            ckpt_dir=ckpt_dir,
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=Dict("encoding" => "byte-mod-vocab", "mode" => "full_model"),
        )
    else
        state = latest === nothing ? DeepSeekHeadTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY) : load_deepseek_checkpoint(latest, model)
        train_deepseek_head_only!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            log_every=1,
            ckpt_dir=ckpt_dir,
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=Dict("encoding" => "byte-mod-vocab", "mode" => "head_only"),
        )
    end

    println("Final loss: $(round(metrics.loss; digits=4))")
    println("Latest checkpoint: $(latest_checkpoint(ckpt_dir))")
end

main()
