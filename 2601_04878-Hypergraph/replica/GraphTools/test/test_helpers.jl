"""
Test helper utilities for GraphTools.jl equivalence testing.
"""
module TestHelpers

using Test
using DataFrames
using Hypergraphs
using Graphs

"""
Compare two graphs for equivalence.
"""
function compare_graphs(g1::AbstractGraph, g2::AbstractGraph)
    if nv(g1) != nv(g2)
        @warn "Vertex count differs: $(nv(g1)) vs $(nv(g2))"
        return false
    end
    
    if ne(g1) != ne(g2)
        @warn "Edge count differs: $(ne(g1)) vs $(ne(g2))"
        return false
    end
    
    # Compare edge sets
    edges1 = Set(edges(g1))
    edges2 = Set(edges(g2))
    
    if edges1 != edges2
        @warn "Edge sets differ"
        return false
    end
    
    return true
end

end # module
