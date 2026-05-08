const _TEB = BytePairEncoding.TextEncodeBase

"""Lightweight metadata about a tokenizer loaded through Julia-native BPE tooling."""
struct NativeTokenizerHandle{E}
    name_or_path::String
    vocab_size::Int
    encoder::E
    special_token_ids::Set{Int}
end

"""
    NativeBPETokenizer(model_id="gpt2")

Tokenizer wrapper backed by Julia-native BytePairEncoding / tiktoken-style
encoders with a zero-based token-id API.
"""
struct NativeBPETokenizer{E}
    model_id::String
    tokenizer::NativeTokenizerHandle{E}
end

function _special_tokens_for_tiktoken(name::AbstractString)
    if name == "gpt2" || name == "r50k_base"
        return Dict("<|endoftext|>" => 50256)
    elseif name == "p50k_base"
        return Dict("<|endoftext|>" => 50256)
    elseif name == "p50k_edit"
        return Dict(
            "<|endoftext|>" => 50256,
            "<|fim_prefix|>" => 50281,
            "<|fim_middle|>" => 50282,
            "<|fim_suffix|>" => 50283,
        )
    elseif name == "cl100k_base"
        return Dict(
            "<|endoftext|>" => 100256,
            "<|fim_prefix|>" => 100257,
            "<|fim_middle|>" => 100258,
            "<|fim_suffix|>" => 100259,
            "<|endofprompt|>" => 100275,
        )
    elseif name == "o200k_base"
        return Dict(
            "<|endoftext|>" => 199998,
            "<|endofprompt|>" => 200017,
        )
    end
    return Dict{String, Int}()
end

function _harmony_special_tokens()
    base = Dict(
        "<|startoftext|>" => 199998,
        "<|endoftext|>" => 199999,
        "<|reserved_200000|>" => 200000,
        "<|reserved_200001|>" => 200001,
        "<|return|>" => 200002,
        "<|constrain|>" => 200003,
        "<|reserved_200004|>" => 200004,
        "<|channel|>" => 200005,
        "<|start|>" => 200006,
        "<|end|>" => 200007,
        "<|message|>" => 200008,
        "<|reserved_200009|>" => 200009,
        "<|reserved_200010|>" => 200010,
        "<|reserved_200011|>" => 200011,
        "<|call|>" => 200012,
    )
    for i in 200013:201087
        base["<|reserved_$(i)|>"] = i
    end
    return base
end

function _load_o200k_harmony_encoder(model_id::String)
    base = BytePairEncoding.load_tiktoken_encoder("o200k_base")
    bpe = base.tokenizer.tokenization.base.bpe
    special_tokens = _harmony_special_tokens()
    matches = collect(keys(special_tokens))
    tokenizer = BytePairEncoding.BPETokenizer(
        _TEB.MatchTokenization(
            BytePairEncoding.BPETokenization(BytePairEncoding.O200kBaseTokenization(), bpe),
            matches,
        ),
    )
    shifted = Dict(token => id + 1 for (token, id) in special_tokens)
    vector = _TEB.PerforatedOverwritableLookupVector(
        _TEB.DATLookupVector(bpe.encoder),
        _TEB.DictBackedLookupDict(shifted, Dict(v => k for (k, v) in shifted)),
    )
    vocab = _TEB.Vocab(vector, "", 0)
    encoder = BytePairEncoding.BPEEncoder(tokenizer, vocab)
    return NativeTokenizerHandle(
        model_id,
        length(vocab),
        encoder,
        Set(values(special_tokens)),
    )
end

function _load_native_tokenizer(model_id::String)
    if model_id in ("gpt2", "openai-community/gpt2")
        encoder = BytePairEncoding.load_tiktoken_encoder("gpt2")
        return NativeTokenizerHandle(model_id, length(encoder.vocab), encoder, Set([50256]))
    elseif model_id in ("r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base")
        encoder = BytePairEncoding.load_tiktoken_encoder(model_id)
        special_ids = Set(values(_special_tokens_for_tiktoken(model_id)))
        return NativeTokenizerHandle(model_id, length(encoder.vocab), encoder, special_ids)
    elseif occursin("gpt-oss", model_id)
        return _load_o200k_harmony_encoder(model_id)
    end
    throw(ArgumentError("unsupported tokenizer model_id=$(model_id); currently supported Julia-native tokenizers are gpt2/openai-community/gpt2, r50k_base, p50k_base, p50k_edit, cl100k_base, o200k_base, and gpt-oss aliases via o200k_harmony"))
end

function NativeBPETokenizer(model_id::String="gpt2")
    handle = _load_native_tokenizer(model_id)
    return NativeBPETokenizer{typeof(handle.encoder)}(model_id, handle)
end

"""Return the vocabulary size of `tok`."""
vocab_size(tok::NativeBPETokenizer) = tok.tokenizer.vocab_size
"""Tokenize `text` with `tok`."""
tokenize(tok::NativeBPETokenizer, text::AbstractString) = encode(tok, text)
"""Decode token ids back into text with `tok`."""
detokenize(tok::NativeBPETokenizer, token_ids::AbstractVector{<:Integer}) = decode(tok, token_ids)

"""Encode `text` into zero-based token ids with the underlying native tokenizer."""
function encode(tok::NativeBPETokenizer, text::AbstractString)
    ids1 = tok.tokenizer.encoder.encode(text)
    return Int.(ids1 .- 1)
end

"""Decode zero-based token ids into text, skipping configured special tokens."""
function decode(tok::NativeBPETokenizer, token_ids::AbstractVector{<:Integer})
    filtered = [Int(id) for id in token_ids if Int(id) ∉ tok.tokenizer.special_token_ids]
    ids1 = Int.(filtered .+ 1)
    return tok.tokenizer.encoder.decode(ids1)
end

"""Return the decoded string for every token in the vocabulary."""
function vocab_texts(tok::NativeBPETokenizer)
    return [tok.tokenizer.encoder.decode([id + 1]) for id in 0:(vocab_size(tok) - 1)]
end

TransformerCore.text_next_token_pairs(texts::AbstractVector{<:AbstractString}, tokenizer::NativeBPETokenizer, seq_len::Integer) =
    TransformerCore.text_next_token_pairs(texts, text -> encode(tokenizer, text), seq_len)
