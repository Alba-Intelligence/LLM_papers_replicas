"""
Test helper utilities for GraphAnalysis.jl equivalence testing.
"""
module TestHelpers

using Test
using DataFrames
using Hypergraphs
using Graphs

"""
Compare two DataFrames for equivalence (same columns, same data).
"""
function compare_dataframes(df1::DataFrame, df2::DataFrame; tolerance::Float64=1e-6)
    if ncol(df1) != ncol(df2)
        @warn "Column count differs: $(ncol(df1)) vs $(ncol(df2))"
        return false
    end
    
    if names(df1) != names(df2)
        @warn "Column names differ: $(names(df1)) vs $(names(df2))"
        return false
    end
    
    if nrow(df1) != nrow(df2)
        @warn "Row count differs: $(nrow(df1)) vs $(nrow(df2))"
        return false
    end
    
    # Compare data values
    for col in names(df1)
        col1 = df1[!, col]
        col2 = df2[!, col]
        
        if eltype(col1) <: AbstractFloat && eltype(col2) <: AbstractFloat
            max_diff = maximum(abs.(col1 .- col2))
            if max_diff > tolerance
                @warn "Column $col differs by more than tolerance: $max_diff > $tolerance"
                return false
            end
        elseif col1 != col2
            @warn "Column $col differs"
            return false
        end
    end
    
    return true
end

end # module
