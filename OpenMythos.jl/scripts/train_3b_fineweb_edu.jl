using Random
using OpenMythos

const TOKENIZER_MODEL_ID = get(
    ENV,
    "OPENMYTHOS_TRAIN_TOKENIZER_MODEL_ID",
    get(ENV, "OPENMYTHOS_TEST_TOKENIZER_MODEL_ID", DEFAULT_MODEL_ID),
)
const USE_FINEWEB = get(ENV, "OPENMYTHOS_USE_FINEWEB_EDU", "0") == "1"
const FINEWEB_SUBSET = get(ENV, "OPENMYTHOS_FINEWEB_SUBSET", "sample-10BT")
const FINEWEB_PARQUET_PATH = get(ENV, "OPENMYTHOS_FINEWEB_PARQUET_PATH", "")
const TRAIN_TEXT_FILE = get(ENV, "OPENMYTHOS_TRAIN_TEXT_FILE", "")
const TRAIN_TEXT = get(ENV, "OPENMYTHOS_TRAIN_TEXT", "")
const CKPT_DIR = get(ENV, "OPENMYTHOS_TRAIN_CKPT_DIR", "checkpoints")
const RAW_TRAIN_MODE = lowercase(get(ENV, "OPENMYTHOS_TRAIN_MODE", "full_model"))
const TRAIN_MODE = RAW_TRAIN_MODE == "full_model" ? "full_model_lux" : RAW_TRAIN_MODE
const TRAIN_ATTN_TYPE = lowercase(get(ENV, "OPENMYTHOS_TRAIN_ATTN_TYPE", "gqa"))
const TRAIN_N_EXPERTS = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_N_EXPERTS", "1"))
const TRAIN_SHARED_EXPERTS = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_SHARED_EXPERTS", "0"))
const TRAIN_EXPERTS_PER_TOKEN = parse(Int, get(ENV, "OPENMYTHOS_TRAIN_EXPERTS_PER_TOKEN", "1"))

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
        if !isempty(FINEWEB_PARQUET_PATH)
            println("Loading FineWeb-Edu batches from local parquet path via Julia...")
            return fineweb_edu_batches_from_parquet(tokenizer, FINEWEB_PARQUET_PATH, SEQ_LEN, BATCH_SIZE; max_batches=FINEWEB_BATCHES)
        end
        println("Loading FineWeb-Edu batches via Python bridge...")
        return fineweb_edu_batches(tokenizer, SEQ_LEN, BATCH_SIZE; subset=FINEWEB_SUBSET, max_batches=FINEWEB_BATCHES)
    end

    texts = _load_local_texts()
    println("Building local text batches...")
    pairs = text_next_token_pairs(texts, tokenizer, SEQ_LEN)
    return batch_next_token_pairs(pairs, BATCH_SIZE; drop_last=false)
end

_legacy_ckpt_dir(train_mode::AbstractString) = joinpath(CKPT_DIR, train_mode)

function main()
    TRAIN_MODE in ("head_only", "full_model_lux", "full_model_legacy") ||
        error("unsupported OPENMYTHOS_TRAIN_MODE=$(RAW_TRAIN_MODE); use head_only, full_model, full_model_lux, or full_model_legacy")

    tokenizer = MythosTokenizer(TOKENIZER_MODEL_ID)
    cfg = TRAIN_MODE in ("full_model_lux", "full_model_legacy") ?
        bootstrap_full_model_training_config(
            OpenMythos.vocab_size(tokenizer);
            seq_len=SEQ_LEN,
            attn_type=TRAIN_ATTN_TYPE,
            n_experts=TRAIN_N_EXPERTS,
            n_shared_experts=TRAIN_SHARED_EXPERTS,
            n_experts_per_tok=TRAIN_EXPERTS_PER_TOKEN,
        ) :
        bootstrap_training_config(OpenMythos.vocab_size(tokenizer); seq_len=SEQ_LEN, attn_type=TRAIN_ATTN_TYPE)
    model = OpenMythos.OpenMythos(cfg; rng=MersenneTwister(RNG_SEED))
    schedule = WarmupCosineSchedule(WARMUP_STEPS, TOTAL_STEPS, LR, LR * 0.1f0)

    batches = _load_batches(tokenizer)
    isempty(batches) && error("no training batches available; provide longer local text or enable FineWeb-Edu")

    latest = if TRAIN_MODE == "full_model_lux"
        latest_checkpoint(CKPT_DIR; family="openmythos", mode="full_model_lux")
    else
        latest_checkpoint(_legacy_ckpt_dir(TRAIN_MODE))
    end

    state = if latest === nothing
        TRAIN_MODE == "full_model_lux" ?
            LuxFullModelTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY, n_loops=cfg.max_loop_iters) :
        TRAIN_MODE == "full_model_legacy" ?
            FullModelTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY, n_loops=cfg.max_loop_iters) :
            HeadOnlyTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY)
    else
        TRAIN_MODE == "full_model_lux" ? load_lux_full_model_checkpoint(latest) :
        TRAIN_MODE == "full_model_legacy" ? load_full_model_checkpoint(latest) :
            load_head_only_checkpoint(latest, model)
    end

    println("Tokenizer model: $(TOKENIZER_MODEL_ID)")
    println("Vocab size: $(OpenMythos.vocab_size(tokenizer)) | seq_len: $(SEQ_LEN) | batch_size: $(BATCH_SIZE) | total_steps: $(TOTAL_STEPS)")
    mode_label = TRAIN_MODE == "full_model_lux" ? "Lux-native full-model bootstrap" :
        TRAIN_MODE == "full_model_legacy" ? "legacy dense full-model bootstrap" :
        "Lux-backed head-only bootstrap"
    attention_details = TRAIN_MODE in ("full_model_lux", "full_model_legacy") ?
        " | routed experts: $(cfg.n_experts) | shared experts: $(cfg.n_shared_experts) | experts/token: $(cfg.n_experts_per_tok)" :
        ""
    println("Training mode: $(mode_label)")
    println("Attention backend: $(TRAIN_ATTN_TYPE)$(attention_details)")
    latest !== nothing && println("Resuming from $(latest)")

    metrics = if TRAIN_MODE == "full_model_lux"
        train_lux_full_model!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            log_every=1,
            ckpt_root=CKPT_DIR,
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=Dict(
                "tokenizer_model_id" => TOKENIZER_MODEL_ID,
                "use_fineweb" => USE_FINEWEB,
                "fineweb_subset" => FINEWEB_SUBSET,
                "fineweb_parquet_path" => FINEWEB_PARQUET_PATH,
            ),
        )
    elseif TRAIN_MODE == "full_model_legacy"
        train_full_model!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            n_loops=cfg.max_loop_iters,
            log_every=1,
            ckpt_dir=_legacy_ckpt_dir(TRAIN_MODE),
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=Dict(
                "tokenizer_model_id" => TOKENIZER_MODEL_ID,
                "use_fineweb" => USE_FINEWEB,
                "fineweb_subset" => FINEWEB_SUBSET,
                "fineweb_parquet_path" => FINEWEB_PARQUET_PATH,
            ),
        )
    else
        train_head_only!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            n_loops=cfg.max_loop_iters,
            log_every=1,
            ckpt_dir=_legacy_ckpt_dir(TRAIN_MODE),
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=Dict(
                "tokenizer_model_id" => TOKENIZER_MODEL_ID,
                "use_fineweb" => USE_FINEWEB,
                "fineweb_subset" => FINEWEB_SUBSET,
                "fineweb_parquet_path" => FINEWEB_PARQUET_PATH,
            ),
        )
    end

    final_latest = TRAIN_MODE == "full_model_lux" ?
        latest_checkpoint(CKPT_DIR; family="openmythos", mode="full_model_lux") :
        latest_checkpoint(_legacy_ckpt_dir(TRAIN_MODE))

    println("Final loss: $(round(metrics.loss; digits=4))")
    println("Latest checkpoint: $(final_latest)")
end

main()
