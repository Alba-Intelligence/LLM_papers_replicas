using Random
using DeepSeekV4
using TextDataCore

const TRAIN_TEXT_FILE = get(ENV, "DEEPSEEK_V4_TRAIN_TEXT_FILE", "")
const TRAIN_TEXT = get(ENV, "DEEPSEEK_V4_TRAIN_TEXT", "")
const TRAIN_MODE = get(ENV, "DEEPSEEK_V4_TRAIN_MODE", "head_only")
const TRAIN_ENCODING = get(ENV, "DEEPSEEK_V4_TRAIN_ENCODING", "tokenizer")
const TOKENIZER_MODEL_ID = get(ENV, "DEEPSEEK_V4_TRAIN_TOKENIZER_MODEL_ID", "gpt2")
const TRAIN_PARQUET_PATH = get(ENV, "DEEPSEEK_V4_TRAIN_PARQUET_PATH", "")
const USE_ENGRAM = get(ENV, "DEEPSEEK_V4_TRAIN_USE_ENGRAM", "0") == "1"
const CKPT_ROOT = get(ENV, "DEEPSEEK_V4_TRAIN_CKPT_DIR", "checkpoints")

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
        "Shared schedules checkpoints and tokenizer-aware text batching now span multiple Julia packages.",
        "This script is a smoke trainer for local text or parquet batches through either tokenizer or byte compatibility modes.",
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

function _load_parquet_texts(path::AbstractString)
    texts = String[]
    for file in parquet_text_files(path)
        append!(texts, parquet_text_column(file))
    end
    return texts
end

_encode_bytes(text::AbstractString, vocab_size::Integer) = Int[mod(Int(b), vocab_size) for b in codeunits(text)]

function _build_tokenizer()
    TRAIN_ENCODING == "tokenizer" || return nothing
    isempty(TOKENIZER_MODEL_ID) && error("DEEPSEEK_V4_TRAIN_TOKENIZER_MODEL_ID must be non-empty when DEEPSEEK_V4_TRAIN_ENCODING=tokenizer")
    return NativeBPETokenizer(TOKENIZER_MODEL_ID)
end

function _resolved_vocab_size(tokenizer)
    return tokenizer === nothing ? VOCAB_SIZE : vocab_size(tokenizer)
end

function _build_engram_lookup(tokenizer)
    USE_ENGRAM || return nothing
    tokenizer === nothing && return nothing
    println("Building Engram token lookup from tokenizer vocabulary surfaces...")
    return build_engram_token_lookup(tokenizer)
end

function _load_batches(tokenizer, vocab_size::Integer)
    if !isempty(TRAIN_PARQUET_PATH)
        if tokenizer !== nothing
            println("Building tokenizer-encoded batches from local parquet text...")
            return next_token_batches_from_parquet(tokenizer, TRAIN_PARQUET_PATH, SEQ_LEN, BATCH_SIZE; max_batches=max(TOTAL_STEPS, 1))
        end
        println("Building byte-encoded batches from local parquet text...")
        texts = _load_parquet_texts(TRAIN_PARQUET_PATH)
        pairs = text_next_token_pairs(texts, text -> _encode_bytes(text, vocab_size), SEQ_LEN)
        return batch_next_token_pairs(pairs, BATCH_SIZE; drop_last=false)
    end

    texts = _load_local_texts()
    if tokenizer !== nothing
        println("Building tokenizer-encoded local-text batches...")
        pairs = text_next_token_pairs(texts, tokenizer, SEQ_LEN)
        return batch_next_token_pairs(pairs, BATCH_SIZE; drop_last=false)
    end

    println("Building local byte-encoded batches...")
    pairs = text_next_token_pairs(texts, text -> _encode_bytes(text, vocab_size), SEQ_LEN)
    return batch_next_token_pairs(pairs, BATCH_SIZE; drop_last=false)
end

function _encoding_details(tokenizer)
    if tokenizer === nothing
        return "byte compatibility | vocab size $(VOCAB_SIZE)"
    end
    return "tokenizer $(tokenizer.model_id) | vocab size $(vocab_size(tokenizer))"
end

function main()
    TRAIN_MODE in ("head_only", "full_model") || error("DEEPSEEK_V4_TRAIN_MODE must be head_only or full_model")
    TRAIN_ENCODING in ("tokenizer", "byte") || error("DEEPSEEK_V4_TRAIN_ENCODING must be tokenizer or byte")

    tokenizer = _build_tokenizer()
    resolved_vocab_size = _resolved_vocab_size(tokenizer)
    engram_lookup = _build_engram_lookup(tokenizer)
    cfg = TRAIN_MODE == "full_model" ?
        bootstrap_deepseek_full_model_training_config(resolved_vocab_size; seq_len=SEQ_LEN, with_engram=USE_ENGRAM, engram_token_lookup=engram_lookup) :
        bootstrap_deepseek_training_config(resolved_vocab_size; seq_len=SEQ_LEN, with_engram=USE_ENGRAM, engram_token_lookup=engram_lookup)
    model = DeepSeekV4Model(cfg; rng=MersenneTwister(RNG_SEED))
    schedule = WarmupCosineSchedule(WARMUP_STEPS, TOTAL_STEPS, LR, LR * 0.1f0)

    batches = _load_batches(tokenizer, cfg.vocab_size)
    isempty(batches) && error("no training batches available; provide longer local text or parquet input")

    latest = latest_checkpoint(CKPT_ROOT; family="deepseekv4", mode=TRAIN_MODE)
    training_mode_label = TRAIN_MODE == "full_model" ? "tiny full-model bootstrap" : "Lux-backed head-only bootstrap"
    engram_label = USE_ENGRAM ? "enabled" : "disabled"

    println("DeepSeek seq_len: $(SEQ_LEN) | batch_size: $(BATCH_SIZE) | total_steps: $(TOTAL_STEPS)")
    println("Encoding: $(_encoding_details(tokenizer))")
    println("Training mode: $(training_mode_label)")
    println("Engram branch: $(engram_label)")
    !isempty(TRAIN_PARQUET_PATH) && println("Parquet path: $(TRAIN_PARQUET_PATH)")
    latest !== nothing && println("Resuming from $(latest)")

    metadata = Dict(
        "encoding" => TRAIN_ENCODING,
        "tokenizer_model_id" => (tokenizer === nothing ? "" : tokenizer.model_id),
        "parquet_path" => TRAIN_PARQUET_PATH,
        "mode" => TRAIN_MODE,
    )

    metrics = if TRAIN_MODE == "full_model"
        state = latest === nothing ? DeepSeekFullModelTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY) : load_deepseek_full_model_checkpoint(latest)
        train_deepseek_full_model!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            log_every=1,
            ckpt_dir=CKPT_ROOT,
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=metadata,
        )
    else
        state = latest === nothing ? DeepSeekHeadTrainerState(model; schedule=schedule, weight_decay=WEIGHT_DECAY) : load_deepseek_checkpoint(latest, model)
        train_deepseek_head_only!(
            state,
            batches;
            total_steps=TOTAL_STEPS,
            log_every=1,
            ckpt_dir=CKPT_ROOT,
            ckpt_every=CKPT_EVERY,
            keep_last=KEEP_LAST,
            checkpoint_metadata=metadata,
        )
    end

    latest_path = latest_checkpoint(CKPT_ROOT; family="deepseekv4", mode=TRAIN_MODE)
    println("Final loss: $(round(metrics.loss; digits=4))")
    println("Latest checkpoint: $(latest_path)")
end

main()
