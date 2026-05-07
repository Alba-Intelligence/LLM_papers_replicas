# GraphGeneration.jl Documentation

```@meta
CurrentModule = GraphGeneration
```

## Overview

GraphGeneration.jl provides functionality to generate hypergraphs from text documents using LLM-based extraction. This package replicates the core functionality from the Python `HyperGraphReasoning` package.

## Installation

```julia
using Pkg
Pkg.add("GraphGeneration")
```

## Core Functions

```@docs
documents2dataframe
recursive_character_text_splitter
hypergraphPrompt
df2hypergraph
make_hypergraph_from_text
```

## Examples

### Basic Usage

```julia
using GraphGeneration

# Convert documents to DataFrame
documents = ["First document", "Second document"]
df = documents2dataframe(documents)

# Generate hypergraph (requires LLM generate function)
# hg, sub_dfs = df2hypergraph(df, generate_function)
```

### Text Chunking

```julia
text = "Long text that needs to be chunked..."
chunks = recursive_character_text_splitter(text; chunk_size=2500, chunk_overlap=0)
```

### Full Workflow

```julia
# Generate hypergraph from text with caching
txt = "Your input text here..."
pkl_path, hg, sub_dfs_path, sub_dfs = make_hypergraph_from_text(
    txt, 
    generate_function,
    "my_graph";
    chunk_size=2500,
    data_dir="data"
)
```

## CLI Usage

```bash
# Generate hypergraph from stdin
echo "Your text" | julia --project=. -e 'using GraphGeneration; GraphGeneration.cli_main()'

# With options
julia --project=. -e 'using GraphGeneration; GraphGeneration.cli_main(["--input", "file.txt", "--output", "output.jld2"])'
```

## Index

```@index
```
