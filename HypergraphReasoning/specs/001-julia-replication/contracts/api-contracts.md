# API Contracts: Julia Replication Project

**Date**: 2026-01-16  
**Purpose**: Define function signatures and interfaces for Julia packages

## Package: GraphGeneration.jl

### Core Functions

#### `documents2dataframe(documents::Vector{String}) -> DataFrame`
Convert vector of text documents to DataFrame with chunk_id.

**Input**:
- `documents`: Vector of text strings

**Output**:
- `DataFrame` with columns: `text`, `chunk_id`

**Errors**:
- Throws `ArgumentError` if documents is empty

---

#### `df2hypergraph(df::DataFrame, generate::Function, ...) -> Tuple{Hypergraph, Vector{DataFrame}}`
Build hypergraph from DataFrame of text chunks.

**Input**:
- `df`: DataFrame with `text` and `chunk_id` columns
- `generate`: Function for LLM text generation (system_prompt, prompt) -> response
- `generate_figure`: Optional function for image analysis
- `image_list`: Optional vector of image paths
- `do_distill`: Boolean, whether to distill text first
- `do_relabel`: Boolean, whether to use relation names as edge IDs
- `repeat_refine`: Integer, number of refinement passes
- `verbatim`: Boolean, whether to print progress

**Output**:
- `Tuple{Hypergraph, Vector{DataFrame}}`: Hypergraph and list of sub-DataFrames (one per chunk)

**Errors**:
- Throws `ArgumentError` if df missing required columns
- Returns `(nothing, DataFrame())` if no valid hypergraphs generated

---

#### `make_hypergraph_from_text(txt::String, generate::Function, ...) -> Tuple{String, Hypergraph, String, Vector{DataFrame}}`
Build or load hypergraph from text, with caching.

**Input**:
- `txt`: Input text string
- `generate`: LLM generation function
- `graph_root`: String identifier for output files
- `chunk_size`: Integer, text chunk size (default: 2500)
- `chunk_overlap`: Integer, overlap between chunks (default: 0)
- `do_distill`: Boolean
- `do_relabel`: Boolean
- `data_dir`: String, output directory
- `verbatim`: Boolean

**Output**:
- `Tuple{String, Hypergraph, String, Vector{DataFrame}}`: (pkl_path, hypergraph, sub_dfs_pkl_path, sub_dfs)

**Errors**:
- Throws `IOError` if data_dir cannot be created
- Returns cached hypergraph if pkl file exists

---

## Package: GraphAnalysis.jl

### Core Functions

#### `find_path(H::Hypergraph, node_embeddings::Dict, tokenizer, model, keyword_1::String, keyword_2::String, ...) -> Tuple`
Find shortest path between two nodes using embeddings.

**Input**:
- `H`: Hypergraph
- `node_embeddings`: Dict mapping node IDs to embedding vectors
- `tokenizer`: Embedding model tokenizer
- `model`: Embedding model
- `keyword_1`, `keyword_2`: Search keywords
- `verbatim`: Boolean
- `second_hop`: Boolean, include 2-hop neighbors
- `data_dir`: String, output directory
- `save_files`: Boolean

**Output**:
- `Tuple` containing: (best_node_1, similarity_1, best_node_2, similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML

**Errors**:
- Returns `nothing` if no path found
- Throws `KeyError` if keywords don't match any nodes

---

#### `s_betweenness_centrality(H::Hypergraph, s::Int=1, edges::Bool=false, normalized::Bool=true) -> Dict`
Compute s-betweenness centrality on hypergraph.

**Input**:
- `H`: Hypergraph
- `s`: Integer, minimum intersection size for s-linegraph
- `edges`: Boolean, compute on edge s-linegraph (true) or node s-linegraph (false)
- `normalized`: Boolean, whether to normalize centrality scores

**Output**:
- `Dict` mapping node/edge IDs to centrality scores (Float64)

**Errors**:
- Throws `ArgumentError` if s < 1

---

#### `is_scale_free(G::AbstractGraph, plot_distribution::Bool=true, data_dir::String="./") -> Tuple{Bool, Fit}`
Determine if graph has scale-free degree distribution.

**Input**:
- `G`: Graph (DiGraph or Hypergraph)
- `plot_distribution`: Boolean, whether to generate plot
- `data_dir`: String, output directory

**Output**:
- `Tuple{Bool, Fit}`: Whether scale-free, and power-law fit object

**Errors**:
- Throws `ArgumentError` if graph is empty

---

## Package: GraphTools.jl

### Core Functions

#### `generate_hypernode_embeddings(nodes, tokenizer, model, embeddings::Dict=Dict(), device::String="cpu") -> Dict`
Generate embeddings for hypergraph nodes.

**Input**:
- `nodes`: Vector of node identifiers or Hypergraph
- `tokenizer`: Embedding model tokenizer (or nothing for sentence-transformers style)
- `model`: Embedding model
- `embeddings`: Existing embeddings dict to update
- `device`: String, compute device ("cpu" or "cuda")

**Output**:
- `Dict` mapping node IDs to embedding vectors (Vector{Float32})

**Errors**:
- Throws `ArgumentError` if model/tokenizer incompatible

---

#### `find_best_fitting_node_list(keyword::String, embeddings::Dict, tokenizer, model, N_samples::Int=5, similarity_threshold::Float64=0.9) -> Vector{Tuple{String, Float64}}`
Find top N nodes matching keyword by embedding similarity.

**Input**:
- `keyword`: Search string
- `embeddings`: Dict of node embeddings
- `tokenizer`: Embedding tokenizer
- `model`: Embedding model
- `N_samples`: Integer, number of results to return
- `similarity_threshold`: Float64, minimum similarity (0.0-1.0)

**Output**:
- `Vector{Tuple{String, Float64}}`: List of (node_id, similarity_score) tuples, sorted descending

**Errors**:
- Returns empty vector if no matches above threshold

---

#### `simplify_hypergraph(H::Hypergraph, sub_dfs::Vector{DataFrame}, node_embeddings::Dict, tokenizer, model, similarity_threshold::Float64=0.9, ...) -> Tuple{Hypergraph, Dict, Vector{DataFrame}}`
Simplify hypergraph by merging similar nodes.

**Input**:
- `H`: Hypergraph to simplify
- `sub_dfs`: Vector of edge DataFrames
- `node_embeddings`: Current node embeddings
- `tokenizer`: Embedding tokenizer
- `model`: Embedding model
- `similarity_threshold`: Float64, cosine similarity threshold for merging (0.0-1.0)
- `data_dir_output`: String
- `graph_root`: String
- `verbatim`: Boolean
- `device`: String

**Output**:
- `Tuple{Hypergraph, Dict, Vector{DataFrame}}`: (simplified_hypergraph, updated_embeddings, updated_sub_dfs)

**Errors**:
- Throws `ArgumentError` if similarity_threshold not in [0, 1]

---

## Package: GraphUtils.jl

### Core Functions

#### `extract(string::String, start::Char='[', end::Char=']') -> String`
Extract substring between delimiters.

**Input**:
- `string`: Input string
- `start`: Start delimiter character
- `end`: End delimiter character

**Output**:
- `String`: Extracted substring including delimiters

**Errors**:
- Returns empty string if delimiters not found

---

#### `remove_markdown_symbols(text::String) -> String`
Remove Markdown formatting from text.

**Input**:
- `text`: String with Markdown formatting

**Output**:
- `String`: Plain text with Markdown removed

---

## CLI Interface Contracts

### GraphGeneration.jl CLI

**Command**: `julia --project=GraphGeneration.jl -e 'using GraphGeneration; GraphGeneration.cli_main()'`

**Arguments**:
- `--input TEXT`: Input text (or read from stdin)
- `--output FILE`: Output file path for hypergraph
- `--format {jld2,json,graphml}`: Output format (default: jld2)
- `--chunk-size INT`: Text chunk size (default: 2500)
- `--chunk-overlap INT`: Chunk overlap (default: 0)
- `--verbose`: Verbose output

**Output**: JSON to stdout with hypergraph metadata, or writes file if `--output` specified

**Errors**: Writes error messages to stderr

---

### GraphAnalysis.jl CLI

**Command**: `julia --project=GraphAnalysis.jl -e 'using GraphAnalysis; GraphAnalysis.cli_main()'`

**Arguments**:
- `--hypergraph FILE`: Input hypergraph file (JLD2/JSON)
- `--embeddings FILE`: Node embeddings file
- `--keyword1 TEXT`: First search keyword
- `--keyword2 TEXT`: Second search keyword
- `--operation {path,centrality,scale-free}`: Analysis operation
- `--output FILE`: Output file path
- `--format {json,graphml}`: Output format

**Output**: JSON to stdout with analysis results

**Errors**: Writes error messages to stderr

---

## Notebook Conversion Contracts

### Conversion Process

**Input**: Python Jupyter notebook (`.ipynb` file) from `HyperGraphReasoningPython/Notebooks/SG/`

**Output**: Pluto notebook (`.jl` file) in `notebooks/SG/` directory

**Conversion Requirements**:
- Translate Python code cells to Julia equivalents
- Replace Python imports with Julia `using` statements (using packages from @replica)
- Convert Python-specific syntax to Julia idioms
- Maintain notebook structure (cells, markdown, code organization)
- Preserve workflow and logic flow
- Update file paths and data loading to use Julia-compatible formats

**Validation**:
- Notebook executes successfully in Pluto.jl
- Outputs match Python notebook outputs (automated comparison where possible, manual verification for interactive/visual outputs)
- All dependencies resolve correctly (packages from @replica available)

**Notebooks to Convert** (MVP scope):
1. `make_hypergraph.ipynb` → `notebooks/SG/make_hypergraph.jl`
2. `Analyze_Hypergraph.ipynb` → `notebooks/SG/Analyze_Hypergraph.jl`
3. `Agents.ipynb` → `notebooks/SG/Agents.jl`

---

## Error Handling

All functions should:
- Use Julia exception types (`ArgumentError`, `KeyError`, `IOError`, etc.)
- Provide descriptive error messages
- Log errors via structured logging (when implemented)
- Return `nothing` or empty collections for "not found" cases (not exceptions)

## Type Contracts

- Hypergraph: `Hypergraphs.Hypergraph`
- Graph: `Graphs.DiGraph` or `Graphs.Graph`
- DataFrame: `DataFrames.DataFrame`
- Embeddings: `Dict{String, Vector{Float32}}`
- LLM Generate Function: `Function` with signature `(system_prompt::String, prompt::String) -> response`
- Pluto Notebook: `.jl` file (Pluto.jl format) stored in `notebooks/` directory
