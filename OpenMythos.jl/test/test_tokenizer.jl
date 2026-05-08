using Test

const TEST_TOKENIZER_MODEL_ID = get(ENV, "OPENMYTHOS_TEST_TOKENIZER_MODEL_ID", "gpt2")

@testset "MythosTokenizer" begin
    tok = MythosTokenizer(TEST_TOKENIZER_MODEL_ID)

    @test tok !== nothing
    @test tok.tokenizer.name_or_path == TEST_TOKENIZER_MODEL_ID
    @test vocab_size(tok) > 0
    @test vocab_size(tok) == tok.tokenizer.vocab_size

    ids = tokenize(tok, "Hello, world!")
    @test ids isa Vector{Int}
    @test !isempty(ids)
    @test all(i -> i isa Int, ids)

    empty_ids = tokenize(tok, "")
    @test empty_ids isa Vector{Int}

    text = detokenize(tok, ids)
    @test text isa String

    original = "The quick brown fox jumps over the lazy dog."
    roundtrip = detokenize(tok, tokenize(tok, original))
    @test occursin(original, roundtrip) || occursin(roundtrip, original)

    long_text = "OpenMythos is a recurrent depth transformer. "^100
    long_ids = tokenize(tok, long_text)
    @test length(long_ids) > 100

    custom = MythosTokenizer(TEST_TOKENIZER_MODEL_ID)
    @test vocab_size(custom) > 0
    @test tokenize(custom, original) == OpenMythos.encode(custom, original)
    @test detokenize(custom, ids) == OpenMythos.decode(custom, ids)

    default_tok = MythosTokenizer()
    @test default_tok.tokenizer.name_or_path == DEFAULT_MODEL_ID
    @test vocab_size(default_tok) > 0

    if occursin("gpt-oss", default_tok.model_id)
        special_ids = tokenize(default_tok, "<|start|>")
        @test !isempty(special_ids)
        @test detokenize(default_tok, special_ids) == ""
    end
end
