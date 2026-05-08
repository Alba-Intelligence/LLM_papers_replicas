using Test
using TextDataCore

@testset "DeepSeek shared tokenizer/data path" begin
    tok = NativeBPETokenizer("gpt2")
    lookup = build_engram_token_lookup(tok)

    @test length(lookup) == vocab_size(tok)
    @test all(id -> id >= 0, lookup)

    mktempdir() do dir
        path = joinpath(dir, "sample.parquet")
        TextDataCore.Parquet2.writefile(
            path,
            (text=[
                "DeepSeek tokenizer-backed parquet smoke data.",
                "A second parquet row keeps the shared data path exercised.",
            ],),
        )

        batches = next_token_batches_from_parquet(tok, path, 4, 1; max_batches=2)
        @test length(batches) == 2
        @test size(batches[1][1]) == (1, 4)
        @test size(batches[1][2]) == (1, 4)
    end
end
