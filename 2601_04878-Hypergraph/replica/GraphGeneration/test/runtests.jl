"""
Test suite for GraphGeneration.jl
Run with: julia --project=. test/runtests.jl
"""
using Test
using Pkg

# Activate package environment
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load package
using GraphGeneration

# Include test files
include("test_documents2dataframe.jl")
include("test_df2hypergraph.jl")
include("test_make_hypergraph.jl")
include("test_integration.jl")
include("test_cli.jl")

println("\n✅ All GraphGeneration.jl tests completed!")
