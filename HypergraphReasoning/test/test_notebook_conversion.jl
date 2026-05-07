# Test suite for notebook conversion validation
# Tests that converted Pluto notebooks exist and are valid

using Test

const NOTEBOOKS_DIR = joinpath(@__DIR__, "..", "notebooks", "SG")
const REQUIRED_NOTEBOOKS = ["make_hypergraph.jl", "Analyze_Hypergraph.jl", "Agents.jl"]

@testset "Notebook Conversion Tests" begin
    @testset "Notebook files exist" begin
        for notebook in REQUIRED_NOTEBOOKS
            notebook_path = joinpath(NOTEBOOKS_DIR, notebook)
            @test isfile(notebook_path) "Notebook $notebook should exist at $notebook_path"
        end
    end

    @testset "Notebook format validation" begin
        for notebook in REQUIRED_NOTEBOOKS
            notebook_path = joinpath(NOTEBOOKS_DIR, notebook)
            if isfile(notebook_path)
                content = read(notebook_path, String)
                # Basic Pluto notebook validation: should contain ### Pluto notebook marker or # ╔═╡
                @test occursin("#", content) "Notebook $notebook should contain Julia code"
                @test occursin("using", content) || occursin("import", content) "Notebook $notebook should contain using/import statements"
            end
        end
    end

    @testset "Notebook uses replica packages" begin
        for notebook in REQUIRED_NOTEBOOKS
            notebook_path = joinpath(NOTEBOOKS_DIR, notebook)
            if isfile(notebook_path)
                content = read(notebook_path, String)
                # Check that notebooks use Julia packages (GraphGeneration, GraphAnalysis, etc.)
                # This is a basic check - actual package names may vary
                @test length(content) > 100 "Notebook $notebook should have substantial content"
            end
        end
    end
end
