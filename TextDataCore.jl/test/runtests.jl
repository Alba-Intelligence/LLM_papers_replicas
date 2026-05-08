using Parquet2
using Test
using TextDataCore

@testset "NativeBPETokenizer" begin
    tok = NativeBPETokenizer("gpt2")
    @test vocab_size(tok) > 0

    ids = tokenize(tok, "Hello, world!")
    @test ids isa Vector{Int}
    @test !isempty(ids)
    @test detokenize(tok, ids) isa String

    surfaces = vocab_texts(tok)
    @test length(surfaces) == vocab_size(tok)
    @test surfaces[1] == "!"
    @test surfaces[end] == "<|endoftext|>"

    harmony = NativeBPETokenizer("gpt-oss-20b")
    @test vocab_size(harmony) > 200000
    @test detokenize(harmony, tokenize(harmony, "<|start|>")) == ""
end

@testset "Parquet text helpers" begin
    tok = NativeBPETokenizer("gpt2")

    mktempdir() do dir
        path = joinpath(dir, "sample.parquet")
        Parquet2.writefile(path, (text=["one short parquet row", "a second parquet row for batching"],))

        @test parquet_text_files(path) == [path]
        @test parquet_text_column(path) == ["one short parquet row", "a second parquet row for batching"]

        batches = next_token_batches_from_parquet(tok, path, 4, 1; max_batches=2)
        @test length(batches) == 2
        @test size(batches[1][1]) == (1, 4)
        @test size(batches[1][2]) == (1, 4)
    end
end
