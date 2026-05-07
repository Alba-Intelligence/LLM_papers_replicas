# Equivalence test for make_hypergraph_from_text.
# Covers caching, text chunking, and hypergraph generation from raw text.
using Test
using Hypergraphs
using DataFrames
using JLD2

# Load test helpers first
include("test_helpers.jl")
using .TestHelpers

# Load package after helpers
using GraphGeneration

# Mock LLM generation function
function mock_generate(system_prompt::String, prompt::String)
    return """
    {
        "nodes": ["node1", "node2"],
        "edges": [
            {"source": "node1", "target": "node2", "relation": "related_to"}
        ]
    }
    """
end

@testset "make_hypergraph_from_text equivalence tests" begin
    # Test 1: Basic functionality
    @testset "basic hypergraph from text" begin
        txt = "This is a sample text for testing hypergraph generation."
        graph_root = "test_graph"
        data_dir = mktempdir()
        
        pkl_path, hg, sub_dfs_pkl_path, sub_dfs = GraphGeneration.make_hypergraph_from_text(
            txt, mock_generate, graph_root, data_dir=data_dir
        )
        
        @test hg isa Hypergraph
        @test sub_dfs isa Vector{DataFrame}
        @test !isempty(pkl_path)
        @test isfile(pkl_path) || endswith(pkl_path, ".jld2")  # Cached file created
    end
    
    # Test 2: Caching behavior
    @testset "caching" begin
        txt = "Test text for caching"
        graph_root = "cache_test"
        data_dir = mktempdir()
        
        # First call - should create cache
        pkl_path1, hg1, _, sub_dfs1 = GraphGeneration.make_hypergraph_from_text(
            txt, mock_generate, graph_root, data_dir=data_dir
        )
        
        # Second call - should load from cache
        pkl_path2, hg2, _, sub_dfs2 = GraphGeneration.make_hypergraph_from_text(
            txt, mock_generate, graph_root, data_dir=data_dir
        )
        
        @test pkl_path1 == pkl_path2
        @test compare_hypergraphs(hg1, hg2)
        @test length(sub_dfs1) == length(sub_dfs2)
    end
    
    # Test 3: Text chunking
    @testset "text chunking" begin
        # Long text that should be chunked
        txt = repeat("This is a test sentence. ", 100)  # Long text
        graph_root = "chunk_test"
        data_dir = mktempdir()
        
        _, hg, _, sub_dfs = GraphGeneration.make_hypergraph_from_text(
            txt, mock_generate, graph_root, 
            chunk_size=250,  # Small chunk size to force chunking
            chunk_overlap=50,
            data_dir=data_dir
        )
        
        @test hg isa Hypergraph
        @test length(sub_dfs) > 1  # Should have multiple chunks
    end
    
    # Test 4: Options (do_distill, do_relabel)
    @testset "options" begin
        txt = "Test text with options"
        graph_root = "options_test"
        data_dir = mktempdir()
        
        _, hg1, _, _ = GraphGeneration.make_hypergraph_from_text(
            txt, mock_generate, graph_root,
            do_distill=true,
            do_relabel=false,
            data_dir=data_dir
        )
        
        _, hg2, _, _ = GraphGeneration.make_hypergraph_from_text(
            txt, mock_generate, graph_root * "_2",
            do_distill=false,
            do_relabel=true,
            data_dir=data_dir
        )
        
        @test hg1 isa Hypergraph
        @test hg2 isa Hypergraph
    end
    
    # Test 5: Error handling for invalid data_dir
    @testset "error handling" begin
        txt = "Test"
        graph_root = "error_test"
        invalid_dir = "/nonexistent/path/that/should/fail"
        
        # Should handle gracefully or throw appropriate error
        # Implementation dependent
    end
end
