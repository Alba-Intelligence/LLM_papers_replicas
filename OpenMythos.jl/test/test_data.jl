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

@testset "FineWeb remote parquet batches via Julia path" begin
    tok = MythosTokenizer("gpt2")

    mktempdir() do dir
        src1 = joinpath(dir, "source1.parquet")
        src2 = joinpath(dir, "source2.parquet")
        OpenMythos.TextDataCore.Parquet2.writefile(src1, (text=["hello world from shard one"],))
        OpenMythos.TextDataCore.Parquet2.writefile(src2, (text=["and shard two carries enough extra tokens for batching"],))

        manifest_calls = NamedTuple[]
        download_calls = Tuple{String, String}[]

        function fake_manifest(; dataset, config, split)
            push!(manifest_calls, (dataset=dataset, config=config, split=split))
            return [src1, src2]
        end

        function fake_downloader(url, path)
            push!(download_calls, (url, path))
            cp(url, path; force=true)
            return path
        end

        batches = fineweb_edu_batches_from_remote_parquet(
            tok,
            2,
            1;
            subset="sample-10BT",
            max_batches=2,
            fetch_manifest=fake_manifest,
            downloader=fake_downloader,
        )

        @test length(batches) == 2
        @test size(batches[1][1]) == (1, 2)
        @test size(batches[1][2]) == (1, 2)
        @test manifest_calls[1].dataset == "HuggingFaceFW/fineweb-edu"
        @test manifest_calls[1].config == "sample-10BT"
        @test manifest_calls[1].split == "train"
        @test length(download_calls) >= 1
    end
end
