using Random
using DeepSeekV4

const TRAIN_TEXT_FILE = get(ENV, "DEEPSEEK_V4_TRAIN_TEXT_FILE", "")
const TRAIN_TEXT = get(ENV, "DEEPSEEK_V4_TRAIN_TEXT", "")
const CKPT_DIR = get(ENV, "DEEPSEEK_V4_TRAIN_CKPT_DIR", "checkpoints")

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
        "DeepSeek V4 in Julia now has a head only bootstrap training surface for tiny configs.",
        "The current training path keeps the DeepSeek body frozen and updates only the language model head.",
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

function main()
    cfg = bootstrap_deepseek_training_config(VOCAB_SIZE; seq_len=SEQ_LEN)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(RNG_SEED))
    schedule = WarmupCosineSchedule(WARMUP_STEPS, TOTAL_STEPS, LR, LR * 0.1f0)

    batches = _load_batches(cfg.vocab_size)
    isempty(batches) && error("no training batches available; provide longer local text input")

    latest = latest_checkpoint(CKPT_DIR)
    state = latest === nothing ? DeepSeekHeadTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY) : load_deepseek_checkpoint(latest, model)

    println("DeepSeek vocab size: $(cfg.vocab_size) | seq_len: $(SEQ_LEN) | batch_size: $(BATCH_SIZE) | total_steps: $(TOTAL_STEPS)")
    println("Training mode: Lux-backed head-only bootstrap")
    latest !== nothing && println("Resuming from $(latest)")

    metrics = train_deepseek_head_only!(
        state,
        batches;
        total_steps=TOTAL_STEPS,
        log_every=1,
        ckpt_dir=CKPT_DIR,
        ckpt_every=CKPT_EVERY,
        keep_last=KEEP_LAST,
        checkpoint_metadata=Dict("encoding" => "byte-mod-vocab"),
    )

    println("Final loss: $(round(metrics.loss; digits=4))")
    println("Latest checkpoint: $(latest_checkpoint(CKPT_DIR))")
end

main()
