"""
Integration test for end-to-end hypergraph generation workflow.
Tests the complete pipeline from text input to hypergraph output.
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

# Mock LLM generation function
function mock_generate(system_prompt::String, prompt::String)
    return """
    {
        "nodes": ["Alice", "Bob", "Charlie"],
        "edges": [
            {"source": "Alice", "target": "Bob", "relation": "knows"},
            {"source": "Bob", "target": "Charlie", "relation": "knows"}
        ]
    }
    """
end

@testset "end-to-end hypergraph generation workflow" begin
    # Test 1: Complete workflow
    @testset "complete workflow" begin
        # Step 1: Convert documents to DataFrame
        documents = [
            "Alice knows Bob. Bob knows Charlie.",
            "Charlie works at a company. The company has many employees."
        ]
        
        df = GraphGeneration.documents2dataframe(documents)
        @test df isa DataFrame
        @test nrow(df) == 2
        
        # Step 2: Convert DataFrame to hypergraph
        hg, sub_dfs = GraphGeneration.df2hypergraph(df, mock_generate)
        @test hg isa Hypergraph
        @test length(sub_dfs) == 2
        
        # Step 3: Verify hypergraph structure
        @test nv(hg) > 0  # Should have nodes
        @test ne(hg) > 0  # Should have edges
    end
    
    # Test 2: Workflow with make_hypergraph_from_text
    @testset "make_hypergraph_from_text workflow" begin
        txt = """
        Alice knows Bob. Bob knows Charlie. 
        Charlie works at a company. The company has many employees.
        Alice and Bob are friends.
        """
        
        graph_root = "integration_test"
        data_dir = mktempdir()
        
        pkl_path, hg, sub_dfs_pkl_path, sub_dfs = GraphGeneration.make_hypergraph_from_text(
            txt, mock_generate, graph_root, data_dir=data_dir
        )
        
        @test hg isa Hypergraph
        @test sub_dfs isa Vector{DataFrame}
        @test !isempty(pkl_path)
        
        # Verify hypergraph has expected structure
        @test nv(hg) > 0
    end
    
    # Test 3: Multiple documents workflow
    @testset "multiple documents" begin
        documents = [
            "First document about science.",
            "Second document about technology.",
            "Third document about innovation."
        ]
        
        df = GraphGeneration.documents2dataframe(documents)
        hg, sub_dfs = GraphGeneration.df2hypergraph(df, mock_generate)
        
        @test length(sub_dfs) == 3
        @test hg isa Hypergraph
    end
    
    # Test 4: Workflow with options
    @testset "workflow with options" begin
        documents = ["Test document"]
        df = GraphGeneration.documents2dataframe(documents)
        
        hg1, _ = GraphGeneration.df2hypergraph(
            df, mock_generate, 
            do_distill=true, 
            do_relabel=false,
            repeat_refine=0
        )
        
        hg2, _ = GraphGeneration.df2hypergraph(
            df, mock_generate,
            do_distill=false,
            do_relabel=true,
            repeat_refine=1
        )
        
        @test hg1 isa Hypergraph
        @test hg2 isa Hypergraph
    end
    
    # Test 5: Error propagation
    @testset "error propagation" begin
        # Empty documents should fail early
        @test_throws ArgumentError GraphGeneration.documents2dataframe(String[])
        
        # Invalid DataFrame should fail in df2hypergraph
        df = DataFrame(text = ["test"])  # Missing chunk_id
        @test_throws ArgumentError GraphGeneration.df2hypergraph(df, mock_generate)
    end
end
