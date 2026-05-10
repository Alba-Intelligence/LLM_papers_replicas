"""
    OLMoTokenizer

OLMo tokenizer wrapper backed by the shared Julia-native
`TextDataCore.NativeBPETokenizer` implementation.

The current OLMo slice defaults to `cl100k_base`, which matches the published
family direction without yet modeling the extra OLMo-specific masking tokens.
"""
struct OLMoTokenizer{E}
    model_id::String
    tokenizer::TextDataCore.NativeTokenizerHandle{E}
end

function OLMoTokenizer(model_id::String=DEFAULT_TOKENIZER_MODEL_ID)
    native = TextDataCore.NativeBPETokenizer(model_id)
    return OLMoTokenizer{typeof(native.tokenizer.encoder)}(native.model_id, native.tokenizer)
end

_native_tokenizer(tok::OLMoTokenizer{E}) where {E} = TextDataCore.NativeBPETokenizer{E}(tok.model_id, tok.tokenizer)

"""Return the vocabulary size of `tok`."""
vocab_size(tok::OLMoTokenizer) = tok.tokenizer.vocab_size
"""Tokenize `text` with `tok`."""
tokenize(tok::OLMoTokenizer, text::AbstractString) = encode(tok, text)
"""Decode token ids back into text with `tok`."""
detokenize(tok::OLMoTokenizer, token_ids::AbstractVector{<:Integer}) = decode(tok, token_ids)

"""Encode `text` into zero-based token ids with the shared native tokenizer."""
encode(tok::OLMoTokenizer, text::AbstractString) = TextDataCore.encode(_native_tokenizer(tok), text)

"""Decode zero-based token ids into text, skipping configured special tokens."""
decode(tok::OLMoTokenizer, token_ids::AbstractVector{<:Integer}) = TextDataCore.decode(_native_tokenizer(tok), token_ids)

TransformerCore.text_next_token_pairs(texts::AbstractVector{<:AbstractString}, tokenizer::OLMoTokenizer, seq_len::Integer) =
    TransformerCore.text_next_token_pairs(texts, text -> encode(tokenizer, text), seq_len)
