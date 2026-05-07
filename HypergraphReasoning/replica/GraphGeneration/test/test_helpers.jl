"""
Test helper utilities for equivalence testing between Julia and Python implementations.
"""
module TestHelpers

using Test
using DataFrames
using Hypergraphs
using Graphs
using JLD2
using JSON

"""
Compare two hypergraphs for equivalence within tolerance.

Returns true if hypergraphs are equivalent (same nodes, edges, structure).
"""
function compare_hypergraphs(hg1::Hypergraph, hg2::Hypergraph; tolerance::Float64=1e-6)
    # Compare node sets
    nodes1 = Set(vertices(hg1))
    nodes2 = Set(vertices(hg2))
    if nodes1 != nodes2
        @warn "Node sets differ: $(setdiff(nodes1, nodes2)) vs $(setdiff(nodes2, nodes1))"
        return false
    end
    
    # Compare edge sets
    edges1 = Set(hyperedges(hg1))
    edges2 = Set(hyperedges(hg2))
    if edges1 != edges2
        @warn "Edge sets differ"
        return false
    end
    
    return true
end

"""
Compare two embedding dictionaries for equivalence within tolerance.

Returns true if embeddings are equivalent (same keys, vectors within tolerance).
"""
function compare_embeddings(emb1::Dict{String, Vector{Float32}}, 
                           emb2::Dict{String, Vector{Float32}}; 
                           tolerance::Float64=1e-5)
    keys1 = Set(keys(emb1))
    keys2 = Set(keys(emb2))
    
    if keys1 != keys2
        @warn "Embedding keys differ: $(setdiff(keys1, keys2)) vs $(setdiff(keys2, keys1))"
        return false
    end
    
    for key in keys1
        v1 = emb1[key]
        v2 = emb2[key]
        
        if length(v1) != length(v2)
            @warn "Embedding dimension mismatch for key $key: $(length(v1)) vs $(length(v2))"
            return false
        end
        
        max_diff = maximum(abs.(v1 .- v2))
        if max_diff > tolerance
            @warn "Embedding difference exceeds tolerance for key $key: $max_diff > $tolerance"
            return false
        end
    end
    
    return true
end

"""
Load reference data from Python output file (JSON or JLD2 format).
"""
function load_reference_data(filepath::String)
    if endswith(filepath, ".json")
        return JSON.parsefile(filepath)
    elseif endswith(filepath, ".jld2")
        return load(filepath)
    else
        error("Unsupported file format: $filepath")
    end
end

"""
Save test output for comparison with Python reference.
"""
function save_test_output(data::Dict, filepath::String)
    if endswith(filepath, ".json")
        open(filepath, "w") do f
            JSON.print(f, data, 2)
        end
    elseif endswith(filepath, ".jld2")
        save(filepath, data)
    else
        error("Unsupported file format: $filepath")
    end
end

end # module
