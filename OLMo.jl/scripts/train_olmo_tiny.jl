using Random
using OLMo

const TOKENIZER_MODEL_ID = get(ENV, "OLMO_TRAIN_TOKENIZER_MODEL_ID", DEFAULT_TOKENIZER_MODEL_ID)
const TRAIN_TEXT_FILE = get(ENV, "OLMO_TRAIN_TEXT_FILE", "")
const TRAIN_TEXT = get(ENV, "OLMO_TRAIN_TEXT", "")
const CKPT_DIR = get(ENV, "OLMO_TRAIN_CKPT_DIR", "checkpoints")
const SEQ_LEN = parse(Int, get(ENV, "OLMO_TRAIN_SEQ_LEN", "32"))
const BATCH_SIZE = parse(Int, get(ENV, "OLMO_TRAIN_BATCH_SIZE", "2"))
const TOTAL_STEPS = parse(Int, get(ENV, "OLMO_TRAIN_TOTAL_STEPS", "8"))
const WARMUP_STEPS = parse(Int, get(ENV, "OLMO_TRAIN_WARMUP_STEPS", "2"))
const LR = parse(Float32, get(ENV, "OLMO_TRAIN_LR", "0.02"))
const WEIGHT_DECAY = parse(Float32, get(ENV, "OLMO_TRAIN_WEIGHT_DECAY", "0.0"))
const CKPT_EVERY = parse(Int, get(ENV, "OLMO_TRAIN_CKPT_EVERY", "4"))
const KEEP_LAST = parse(Int, get(ENV, "OLMO_TRAIN_KEEP_LAST", "3"))
const RNG_SEED = parse(Int, get(ENV, "OLMO_TRAIN_SEED", "1"))

function _default_texts()
    return [
        "OLMo 2 in Julia keeps the dense decoder stack clear and small.",
        "This bootstrap trainer is intentionally tiny and smoke-test oriented.",
        "The first OLMo slice focuses on post norm MHA and QK norm.",
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

function main()
    tokenizer = OLMoTokenizer(TOKENIZER_MODEL_ID)
    cfg = bootstrap_olmo_training_config(vocab_size(tokenizer); seq_len=SEQ_LEN)
    model = OLMoModel(cfg; rng=MersenneTwister(RNG_SEED))
    schedule = WarmupCosineSchedule(WARMUP_STEPS, TOTAL_STEPS, LR, LR * 0.1f0)

    texts = _load_local_texts()
    pairs = text_next_token_pairs(texts, tokenizer, SEQ_LEN)
    batches = batch_next_token_pairs(pairs, BATCH_SIZE; drop_last=false)
    isempty(batches) && error("no training batches available; provide longer local text")

    latest = latest_checkpoint(CKPT_DIR; family="olmo", mode="full_model")
    state = latest === nothing ?
        OLMoFullModelTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY) :
        load_olmo_checkpoint(latest)

    println("Tokenizer model: $(TOKENIZER_MODEL_ID)")
    println("Vocab size: $(vocab_size(tokenizer)) | seq_len: $(SEQ_LEN) | batch_size: $(BATCH_SIZE) | total_steps: $(TOTAL_STEPS)")
    latest !== nothing && println("Resuming from $(latest)")

    metrics = train_olmo!(
        state,
        batches;
        total_steps=TOTAL_STEPS,
        log_every=1,
        ckpt_root=CKPT_DIR,
        ckpt_every=CKPT_EVERY,
        keep_last=KEEP_LAST,
        checkpoint_metadata=Dict("tokenizer_model_id" => TOKENIZER_MODEL_ID),
    )

    final_latest = latest_checkpoint(CKPT_DIR; family="olmo", mode="full_model")
    println("Final loss: $(round(metrics.loss; digits=4))")
    println("Latest checkpoint: $(final_latest)")
end

main()
