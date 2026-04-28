using Test

const TEST_TOKENIZER_MODEL_ID = get(ENV, "OPENMYTHOS_TEST_TOKENIZER_MODEL_ID", "gpt2")

@testset "MythosTokenizer" begin
    tok = MythosTokenizer(TEST_TOKENIZER_MODEL_ID)

    @test tok !== nothing
    @test tok.tokenizer.name_or_path == TEST_TOKENIZER_MODEL_ID
    @test OpenMythos.vocab_size(tok) > 0
    @test OpenMythos.vocab_size(tok) == tok.tokenizer.vocab_size

    ids = OpenMythos.encode(tok, "Hello, world!")
    @test ids isa Vector{Int}
    @test !isempty(ids)
    @test all(i -> i isa Int, ids)

    empty_ids = OpenMythos.encode(tok, "")
    @test empty_ids isa Vector{Int}

    text = OpenMythos.decode(tok, ids)
    @test text isa String

    original = "The quick brown fox jumps over the lazy dog."
    roundtrip = OpenMythos.decode(tok, OpenMythos.encode(tok, original))
    @test occursin(original, roundtrip) || occursin(roundtrip, original)

    long_text = "OpenMythos is a recurrent depth transformer. "^100
    long_ids = OpenMythos.encode(tok, long_text)
    @test length(long_ids) > 100

    custom = MythosTokenizer(TEST_TOKENIZER_MODEL_ID)
    @test OpenMythos.vocab_size(custom) > 0
end
