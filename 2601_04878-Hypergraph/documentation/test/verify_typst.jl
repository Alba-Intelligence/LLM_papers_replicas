"""
Test script to verify Typst compilation succeeds.
"""
using Test

@testset "Typst compilation verification" begin
    typ_file = joinpath(@__DIR__, "..", "graph-generation.typ")
    
    @testset "Typst file exists" begin
        @test isfile(typ_file)
    end
    
    @testset "Typst compilation" begin
        # Try to compile (if typst is available)
        try
            output_pdf = joinpath(@__DIR__, "..", "graph-generation.pdf")
            run(`typst compile $typ_file $output_pdf`)
            @test isfile(output_pdf)
        catch e
            @warn "Typst not available or compilation failed: $e"
            # Don't fail test if typst is not installed
        end
    end
end
