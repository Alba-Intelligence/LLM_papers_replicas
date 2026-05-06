"""
Equivalence test for documents2dataframe function.
Tests that Julia implementation produces same output as Python reference.
"""
using Test
using DataFrames
using JSON

# Load test helpers first
include("test_helpers.jl")
using .TestHelpers

# Load package after helpers
using GraphGeneration

@testset "documents2dataframe equivalence tests" begin
    # Test 1: Basic functionality
    @testset "basic conversion" begin
        documents = [
            "This is the first document.",
            "This is the second document with more content.",
            "A third document for testing."
        ]
        
        df = GraphGeneration.documents2dataframe(documents)
        
        @test df isa DataFrame
        @test ncol(df) == 2
        @test names(df) == ["text", "chunk_id"]
        @test nrow(df) == 3
        @test df.text == documents
    end
    
    # Test 2: Equivalence with Python reference
    @testset "equivalence with Python reference" begin
        # Load Python reference if available
        ref_path = joinpath(@__DIR__, "fixtures", "documents2dataframe_reference.json")
        
        if isfile(ref_path)
            ref_data = load_reference_data(ref_path)
            documents = ref_data["text"]
            
            df = GraphGeneration.documents2dataframe(documents)
            
            # Compare with reference
            @test df.text == ref_data["text"]
            @test df.chunk_id == ref_data["chunk_id"]
        else
            @warn "Reference file not found: $ref_path. Skipping equivalence test."
        end
    end
    
    # Test 3: Empty input error
    @testset "empty input error" begin
        @test_throws ArgumentError GraphGeneration.documents2dataframe(String[])
    end
    
    # Test 4: Single document
    @testset "single document" begin
        documents = ["Single document test"]
        df = GraphGeneration.documents2dataframe(documents)
        
        @test nrow(df) == 1
        @test df.text[1] == "Single document test"
        @test !isempty(df.chunk_id[1])
    end
    
    # Test 5: Chunk ID consistency (same text = same chunk_id)
    @testset "chunk ID consistency" begin
        documents = ["Same text", "Same text", "Different text"]
        df = GraphGeneration.documents2dataframe(documents)
        
        @test df.chunk_id[1] == df.chunk_id[2]  # Same text should have same chunk_id
        @test df.chunk_id[1] != df.chunk_id[3]   # Different text should have different chunk_id
    end
end
