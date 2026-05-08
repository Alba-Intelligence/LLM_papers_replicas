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
        OpenMythos.TextDataCore.Parquet2.writefile(path, tbl)

        batches = fineweb_edu_batches_from_parquet(tok, path, 4, 1; max_batches=2)
        @test length(batches) == 2
        @test size(batches[1][1]) == (1, 4)
        @test size(batches[1][2]) == (1, 4)
    end
end

@testset "FineWeb rows batches via Julia API path" begin
    tok = MythosTokenizer("gpt2")
    calls = NamedTuple[]
    pages = Dict(
        0 => [
            "OpenMythos remote rows smoke data one.",
            "OpenMythos remote rows smoke data two.",
        ],
    )

    function fake_rows_fetch(; dataset, config, split, offset, length)
        push!(calls, (dataset=dataset, config=config, split=split, offset=offset, length=length))
        return get(pages, offset, String[])
    end

    batches = fineweb_edu_batches(tok, 4, 1; subset="sample-10BT", max_batches=2, fetch_rows=fake_rows_fetch)
    @test length(batches) == 2
    @test size(batches[1][1]) == (1, 4)
    @test size(batches[1][2]) == (1, 4)
    @test calls[1].dataset == "HuggingFaceFW/fineweb-edu"
    @test calls[1].config == "sample-10BT"
    @test calls[1].split == "train"
    @test calls[1].offset == 0
end
