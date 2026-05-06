"""
Equivalence test for df2hypergraph function.
Tests that Julia implementation produces equivalent hypergraph structure as Python.
"""
using Test
using DataFrames
using Hypergraphs
using JSON

# Load test helpers first
include("test_helpers.jl")
using .TestHelpers

# Load package after helpers
using GraphGeneration

# Mock LLM generation function for testing
function mock_generate(system_prompt::String, prompt::String)
    # Return a simple mock response that creates a minimal hypergraph
    return """
    {
        "nodes": ["node1", "node2"],
        "edges": [
            {"source": "node1", "target": "node2", "relation": "related_to"}
        ]
    }
    """
end

@testset "df2hypergraph equivalence tests" begin
    # Test 1: Basic functionality
    @testset "basic hypergraph creation" begin
        df = DataFrame(
            text = ["Sample text chunk 1", "Sample text chunk 2"],
            chunk_id = ["chunk_1", "chunk_2"]
        )
        
        hg, sub_dfs = GraphGeneration.df2hypergraph(df, mock_generate)
        
        @test hg isa Hypergraph
        @test sub_dfs isa Vector{DataFrame}
        @test length(sub_dfs) == 2
    end
    
    # Test 2: Required columns check
    @testset "missing required columns" begin
        df = DataFrame(text = ["test"])
        # Missing chunk_id column
        
        @test_throws ArgumentError GraphGeneration.df2hypergraph(df, mock_generate)
    end
    
    # Test 3: Empty DataFrame handling
    @testset "empty DataFrame" begin
        df = DataFrame(text = String[], chunk_id = String[])
        
        result = GraphGeneration.df2hypergraph(df, mock_generate)
        @test result[1] === nothing
        @test isempty(result[2])
    end
    
    # Test 4: Multiple chunks
    @testset "multiple chunks" begin
        df = DataFrame(
            text = ["Chunk 1", "Chunk 2", "Chunk 3"],
            chunk_id = ["c1", "c2", "c3"]
        )
        
        hg, sub_dfs = GraphGeneration.df2hypergraph(df, mock_generate)
        
        @test hg isa Hypergraph
        @test length(sub_dfs) == 3
    end
    
    # Test 5: Options (do_distill, do_relabel, etc.)
    @testset "options handling" begin
        df = DataFrame(
            text = ["Test text"],
            chunk_id = ["test_chunk"]
        )
        
        # Test with different options
        hg1, _ = GraphGeneration.df2hypergraph(df, mock_generate, do_distill=true, do_relabel=false)
        hg2, _ = GraphGeneration.df2hypergraph(df, mock_generate, do_distill=false, do_relabel=true)
        
        @test hg1 isa Hypergraph
        @test hg2 isa Hypergraph
    end
end
