using Random
using OpenMythos

const TOKENIZER_MODEL_ID = get(
    ENV,
    "OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID",
    get(ENV, "OPENMYTHOS_TEST_TOKENIZER_MODEL_ID", DEFAULT_MODEL_ID),
)
const USE_FINEWEB = get(ENV, "OPENMYTHOS_USE_FINEWEB_EDU", "0") == "1"
const FINEWEB_SUBSET = get(ENV, "OPENMYTHOS_FINEWEB_SUBSET", "sample-10BT")
const TRAIN_TEXT_FILE = get(ENV, "OPENMYTHOS_TRAIN_TEXT_FILE", "")
const TRAIN_TEXT = get(ENV, "OPENMYTHOS_TRAIN_TEXT", "")
const CKPT_DIR = get(ENV, "OPENMYTHOS_TRAIN_CKPT_DIR", "checkpoints")
const TRAIN_MODE = lowercase(get(ENV, "OPENMYTHOS_TRAIN_MODE", "head_only"))
const TRAIN_ATTN_TYPE = lowercase(get(ENV, "OPENMYTHOS_TRAIN_ATTN_TYPE", "gqa"))
const TRAIN_SHARED_EXPERTS = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_SHARED_EXPERTS", "0"))

const SEQ_LEN = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_SEQ_LEN", "32"))
const BATCH_SIZE = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_BATCH_SIZE", "2"))
const TOTAL_STEPS = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_TOTAL_STEPS", "8"))
const WARMUP_STEPS = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_WARMUP_STEPS", "2"))
const LR = parse(Float32, get(ENV, "OPENMYTHOS_TRAIN_LR", "0.02"))
const WEIGHT_DECAY = parse(Float32, get(ENV, "OPENMYTHOS_TRAIN_WEIGHT_DECAY", "0.0"))
const CKPT_EVERY = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_CKPT_EVERY", "4"))
const KEEP_LAST = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_KEEP_LAST", "3"))
const FINEWEB_BATCHES = parse(Int, get(ENV, "OPENMYTHOS_FINEWEB_BATCHES", string(max(TOTAL_STEPS, 1))))
const RNG_SEED = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_SEED", "1"))

function _default_texts()
    return [
        "OpenMythos in Julia keeps the recurrent depth transformer architecture intact.",
        "Bootstrap training currently updates the language-model head while the rest of the port stays parity-focused.",
        "FineWeb-Edu integration is available through an optional Python streaming bridge for small smoke runs.",
        "Tokenizer batching checkpointing and learning-rate scheduling now exist in the Julia replica.",
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

function _load_batches(tokenizer::MythosTokenizer)
    if USE_FINEWEB
        println("Loading FineWeb-Edu batches via Python bridge...")
        return fineweb_edu_batches(tokenizer, SEQ_LEN, BATCH_SIZE; subset=FINEWEB_SUBSET, max_batches=FINEWEB_BATCHES)
    end

    texts = _load_local_texts()
    println("Building local text batches...")
    pairs = text_next_token_pairs(texts, tokenizer, SEQ_LEN)
    return batch_next_token_pairs(pairs, BATCH_SIZE; drop_last=false)
end

function main()
    tokenizer = MythosTokenizer(TOKENIZER_MODEL_ID)
    cfg = TRAIN_MODE == "full_model" ?
        bootstrap_full_model_training_config(
            OpenMythos.vocab_size(tokenizer);
            seq_len=SEQ_LEN,
            attn_type=TRAIN_ATTN_TYPE,
            n_shared_experts=TRAIN_SHARED_EXPERTS,
        ) :
        bootstrap_training_config(OpenMythos.vocab_size(tokenizer); seq_len=SEQ_LEN, attn_type=TRAIN_ATTN_TYPE)
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(RNG_SEED))
    schedule = WarmupCosineSchedule(WARMUP_STEPS, TOTAL_STEPS, LR, LR * 0.1f0)

    batches = _load_batches(tokenizer)
    isempty(batches) && error("no training batches available; provide longer local text or enable FineWeb-Edu")

    latest = latest_checkpoint(CKPT_DIR)
    state = if latest === nothing
        TRAIN_MODE == "full_model" ?
            FullModelTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY, n_loops=cfg.max_loop_iters) :
            HeadOnlyTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY)
    else
        TRAIN_MODE == "full_model" ? load_full_model_checkpoint(latest) : load_head_only_checkpoint(latest, model)
    end

    println("Tokenizer model: $(TOKENIZER_MODEL_ID)")
    println("Vocab size: $(OpenMythos.vocab_size(tokenizer)) | seq_len: $(SEQ_LEN) | batch_size: $(BATCH_SIZE) | total_steps: $(TOTAL_STEPS)")
    println("Training mode: $(TRAIN_MODE == \"full_model\" ? \"dense full-model bootstrap\" : \"Lux-backed head-only bootstrap\")")
    println("Attention backend: $(TRAIN_ATTN_TYPE)$(TRAIN_MODE == \"full_model\" ? \" | shared experts: $(cfg.n_shared_experts)\" : \"\")")
    latest !== nothing && println("Resuming from $(latest)")

    metrics = if TRAIN_MODE == "full_model"
        train_full_model!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            n_loops=cfg.max_loop_iters,
            log_every=1,
            ckpt_dir=CKPT_DIR,
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=Dict(
                "tokenizer_model_id" => TOKENIZER_MODEL_ID,
                "use_fineweb" => USE_FINEWEB,
                "fineweb_subset" => FINEWEB_SUBSET,
            ),
        )
    else
        train_head_only!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            n_loops=cfg.max_loop_iters,
            log_every=1,
            ckpt_dir=CKPT_DIR,
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=Dict(
                "tokenizer_model_id" => TOKENIZER_MODEL_ID,
                "use_fineweb" => USE_FINEWEB,
                "fineweb_subset" => FINEWEB_SUBSET,
            ),
        )
    end

    println("Final loss: $(round(metrics.loss; digits=4))")
    println("Latest checkpoint: $(latest_checkpoint(CKPT_DIR))")
end

main()
