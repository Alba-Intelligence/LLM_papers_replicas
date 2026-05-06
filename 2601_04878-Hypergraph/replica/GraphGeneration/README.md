# GraphGeneration.jl

Generate hypergraphs from text documents using LLM-based extraction.

## Installation

```julia
using Pkg
Pkg.add("GraphGeneration")
```

Or from local path:

```julia
Pkg.add(path="replica/GraphGeneration")
```

## Quick Start

```julia
using GraphGeneration

# Convert documents to DataFrame
documents = ["First document", "Second document"]
df = documents2dataframe(documents)

# Generate hypergraph (requires LLM generate function)
function my_generate(system_prompt, prompt)
    # Your LLM integration here
    return """{"nodes": [], "edges": []}"""
end

hg, sub_dfs = df2hypergraph(df, my_generate)
```

## CLI Usage

```bash
# From stdin
echo "Your text here" | julia --project=. -e 'using GraphGeneration; GraphGeneration.cli_main()'

# With file input
julia --project=. -e 'using GraphGeneration; GraphGeneration.cli_main(["--input", "input.txt", "--output", "output.jld2"])'
```

## Documentation

Full API documentation available at: `docs/build/` (build with `julia docs/make.jl`)

## Development

Run tests:

```julia
using Pkg
Pkg.test("GraphGeneration")
```

## License

See repository LICENSE file.
