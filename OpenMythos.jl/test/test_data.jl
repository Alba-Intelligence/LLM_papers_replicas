using Parquet2
using Test

@testset "FineWeb parquet batches via Julia path" begin
    tok = MythosTokenizer("gpt2")

    mktempdir() do dir
        path = joinpath(dir, "sample.parquet")
        tbl = (
            text=[
                "OpenMythos Julia parquet smoke data for tokenizer batching.",
                "A second row keeps the local parquet FineWeb replacement path exercised.",
            ],
        )
        Parquet2.writefile(path, tbl)

        batches = fineweb_edu_batches_from_parquet(tok, path, 4, 1; max_batches=2)
        @test length(batches) == 2
        @test size(batches[1][1]) == (1, 4)
        @test size(batches[1][2]) == (1, 4)
    end
end
