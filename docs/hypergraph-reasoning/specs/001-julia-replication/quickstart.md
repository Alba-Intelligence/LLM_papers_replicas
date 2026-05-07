# Quickstart Guide: Julia Replication Project

**Date**: 2026-01-16  
**Purpose**: Get started with Julia package replication

## Prerequisites

- Julia 1.9+ installed
- Python 3.10+ (for reference implementation and testing)
- Git

## Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd 2601_04878-Hypergraph
git checkout 001-julia-replication
```

### 2. Create Julia Environment

```bash
# Create main project environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 3. Create First Julia Package

```bash
# Create GraphGeneration.jl package
cd replica
julia -e 'using Pkg; Pkg.generate("GraphGeneration")'
cd GraphGeneration
```

### 4. Add Dependencies

Edit `Project.toml` in `GraphGeneration.jl/`:

```toml
[deps]
Graphs = "86223c79-3864-5bf0-83f7-82e325a16860"
Hypergraphs = "2d0c8e7d-4b0e-4b0f-8a0e-4b0f-8a0e-4b0f-8a0e"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
ArgParse = "c7e460c6-2fb9-53a9-8c5b-16f535851c63"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
JLD2 = "033835bb-8acc-5ee8-8a36-4e1a48c6bbf0"
```

Then:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Development Workflow

### 1. Write Tests First (TDD)

Create `test/runtests.jl`:

```julia
using Test
using GraphGeneration

@testset "GraphGeneration.jl" begin
    # Write failing tests first
    @test documents2dataframe(["test"]) isa DataFrame
end
```

Run tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### 2. Implement Function

Edit `src/GraphGeneration.jl`:

```julia
module GraphGeneration

using DataFrames
using SHA

export documents2dataframe

function documents2dataframe(documents::Vector{String})::DataFrame
    rows = []
    for chunk in documents
        chunk_id = bytes2hex(sha256(chunk))
        push!(rows, (text=chunk, chunk_id=chunk_id))
    end
    return DataFrame(rows)
end

end # module
```

### 3. Run Tests Again

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests should now pass.

### 4. Add CLI Interface

Add to `src/GraphGeneration.jl`:

```julia
using ArgParse

function cli_main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input"
            help = "Input text (or read from stdin)"
            arg_type = String
            default = ""
        "--output"
            help = "Output file path"
            arg_type = String
        "--format"
            help = "Output format (jld2, json, graphml)"
            arg_type = String
            default = "jld2"
    end
    
    args = parse_args(s)
    
    # Read input
    input_text = if args["input"] == ""
        read(stdin, String)
    else
        read(args["input"], String)
    end
    
    # Process (placeholder)
    result = Dict("status" => "ok", "input_length" => length(input_text))
    
    # Output
    if haskey(args, "output")
        open(args["output"], "w") do f
            JSON.print(f, result)
        end
    else
        JSON.print(stdout, result)
    end
end
```

### 5. Test CLI

```bash
echo "test input" | julia --project=. -e 'using GraphGeneration; GraphGeneration.cli_main()' --input ""
```

## Documentation

### Documenter.jl Setup

Create `docs/make.jl`:

```julia
using Documenter
using GraphGeneration

makedocs(
    sitename = "GraphGeneration",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ]
)
```

Create `docs/src/index.md`:

```markdown
# GraphGeneration.jl

Package for generating hypergraphs from text.

## Installation

```julia
using Pkg
Pkg.add("GraphGeneration")
```

## Quick Example

```julia
using GraphGeneration

docs = ["First chunk", "Second chunk"]
df = documents2dataframe(docs)
```
```

Build docs:

```bash
julia docs/make.jl
```

### Typst Documentation

Create `docs/hypergraph-reasoning/graph-generation.typ`:

```typst
#set page(margin: 2cm)
#set text(size: 11pt)

= Graph Generation Module Replication

== Progress

Started: 2026-01-16

Status: In progress

== Implementation Approach

We are replicating `graph_generation.py` functionality...

== Learnings

* Julia DataFrames.jl provides similar API to pandas
* Hypergraphs.jl requires different API than HyperNetX
```

Compile:

```bash
typst compile docs/hypergraph-reasoning/graph-generation.typ
```

## Testing Equivalence

### Generate Reference Data

```python
# In Python environment
from GraphReasoning.graph_generation import documents2Dataframe
import pandas as pd
import json

docs = ["test chunk 1", "test chunk 2"]
df = documents2Dataframe(docs)
df.to_json("test/fixtures/reference_documents2dataframe.json", orient="records")
```

### Compare in Julia Tests

```julia
using Test
using JSON
using GraphGeneration

@testset "Equivalence: documents2dataframe" begin
    # Load reference
    ref = JSON.parsefile("test/fixtures/reference_documents2dataframe.json")
    
    # Run Julia version
    docs = ["test chunk 1", "test chunk 2"]
    result = documents2dataframe(docs)
    
    # Compare
    @test length(result) == length(ref)
    @test result.text == [r["text"] for r in ref]
    @test result.chunk_id == [r["chunk_id"] for r in ref]
end
```

## Notebook Conversion

### 1. Install Pluto.jl

```julia
using Pkg
Pkg.add("Pluto")
```

### 2. Convert Python Notebook

Manual conversion process:

1. **Read Python notebook**: Open `.ipynb` file and identify cells
2. **Translate code**: Convert Python code to Julia equivalents
   - Replace `import` with `using`
   - Update function calls to Julia syntax
   - Convert data structures (dict → Dict, list → Vector, etc.)
3. **Update dependencies**: Replace Python imports with Julia `using` statements:
   ```julia
   # Python: from GraphReasoning import graph_generation
   # Julia:
   using GraphGeneration  # from the top-level Julia package
   ```
4. **Create Pluto notebook**: Save as `.jl` file in `notebooks/hypergraph-reasoning/sg/` directory
5. **Test execution**: Open in Pluto.jl and verify it runs

### 3. Example Conversion

**Python (Jupyter)**:
```python
import pandas as pd
from GraphReasoning.graph_generation import documents2Dataframe

docs = ["First document", "Second document"]
df = documents2Dataframe(docs)
print(df)
```

**Julia (Pluto)**:
```julia
### A Pluto.jl notebook ###
# ╔═╡ Cell order:
# ╠═[cell 1]
# ╠═[cell 2]

# ╔═╡ [cell 1]
using DataFrames
using GraphGeneration

# ╔═╡ [cell 2]
docs = ["First document", "Second document"]
df = documents2dataframe(docs)
df
```

### 4. Validate Conversion

```julia
# In Pluto notebook or test script
using Pluto
using Test

# Execute notebook and compare outputs
# Automated: Extract outputs and compare with Python reference
# Manual: Execute in Pluto and visually verify results
```

## Next Steps

1. Implement core functions from `graph_generation.py`
2. Convert Python notebooks to Pluto notebooks
3. Add integration tests for inter-package dependencies
4. Update AGENTS.md with Julia-specific conventions
5. Document design decisions in Typst files

## Resources

- [Julia Documentation](https://docs.julialang.org/)
- [Graphs.jl Documentation](https://github.com/JuliaGraphs/Graphs.jl)
- [Hypergraphs.jl Documentation](https://github.com/JuliaGraphs/Hypergraphs.jl)
- [DataFrames.jl Documentation](https://dataframes.juliadata.org/)
- [Pluto.jl Documentation](https://plutojl.org/)
