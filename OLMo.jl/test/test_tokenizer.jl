@testset "OLMo tokenizer wrapper" begin
    tok = OLMoTokenizer("cl100k_base")
    text = "OLMo in Julia"
    ids = encode(tok, text)

    @test !isempty(ids)
    @test decode(tok, ids) == text
    @test vocab_size(tok) > 100_000
end
