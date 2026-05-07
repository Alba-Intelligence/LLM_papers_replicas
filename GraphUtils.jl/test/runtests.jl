# Test suite for GraphUtils.jl.
# Run with: julia --project=. test/runtests.jl
using Test
using Pkg

# Activate package environment
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load package
using GraphUtils

# Include test files
include("test_cli.jl")

println("\n✅ All GraphUtils.jl tests completed!")
