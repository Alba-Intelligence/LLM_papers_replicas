"""
    MythosTokenizer

OpenMythos tokenizer wrapper backed by the shared Julia-native
`TextDataCore.NativeBPETokenizer` implementation.
"""
struct MythosTokenizer{E}
    model_id::String
    tokenizer::TextDataCore.NativeTokenizerHandle{E}
end

function MythosTokenizer(model_id::String=DEFAULT_MODEL_ID)
    native = TextDataCore.NativeBPETokenizer(model_id)
    return MythosTokenizer{typeof(native.tokenizer.encoder)}(native.model_id, native.tokenizer)
end

_native_tokenizer(tok::MythosTokenizer{E}) where {E} = TextDataCore.NativeBPETokenizer{E}(tok.model_id, tok.tokenizer)

"""Return the vocabulary size of `tok`."""
vocab_size(tok::MythosTokenizer) = tok.tokenizer.vocab_size
"""Tokenize `text` with `tok`."""
tokenize(tok::MythosTokenizer, text::AbstractString) = encode(tok, text)
"""Decode token ids back into text with `tok`."""
detokenize(tok::MythosTokenizer, token_ids::AbstractVector{<:Integer}) = decode(tok, token_ids)

"""Encode `text` into zero-based token ids with the shared native tokenizer."""
encode(tok::MythosTokenizer, text::AbstractString) = TextDataCore.encode(_native_tokenizer(tok), text)

"""Decode zero-based token ids into text, skipping configured special tokens."""
decode(tok::MythosTokenizer, token_ids::AbstractVector{<:Integer}) = TextDataCore.decode(_native_tokenizer(tok), token_ids)
