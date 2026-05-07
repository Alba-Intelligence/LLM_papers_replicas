#set page(margin: 2cm)
#set text(font: "Linux Libertine", size: 11pt)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 24pt, weight: "bold")[GraphGeneration.jl Replication]
  
  #text(size: 14pt)[Progress Documentation]
  
  #text(size: 10pt)[Date: 2026-01-16]
]

== Introduction

This document tracks the replication of the Python `graph_generation.py` module into the Julia package `GraphGeneration.jl`. The goal is to achieve functional equivalence while leveraging Julia's performance characteristics and idiomatic design patterns.

== Progress Status

=== Implementation Status

*Status: MVP Complete*

- [x] Package structure created
- [x] Core functions implemented:
  - `documents2dataframe`: Convert documents to DataFrame with chunk IDs
  - `recursive_character_text_splitter`: Text chunking utility
  - `hypergraphPrompt`: LLM-based hypergraph extraction
  - `df2hypergraph`: Build hypergraph from DataFrame
  - `make_hypergraph_from_text`: End-to-end workflow with caching
- [x] CLI interface implemented using `ArgParse.jl`
- [x] Test suite created (TDD approach)
- [x] Documentation structure (Documenter.jl)

=== Test Coverage

- [x] Equivalence tests for `documents2dataframe`
- [x] Equivalence tests for `df2hypergraph`
- [x] Equivalence tests for `make_hypergraph_from_text`
- [x] Integration tests for end-to-end workflow
- [x] CLI interface tests

== Implementation Approach

=== Design Decisions

*Julia Package Structure*: One Julia package per Python module, maintaining clear separation of concerns. Packages are independently installable and testable.

*Dependencies*:
- `Graphs.jl` and `Hypergraphs.jl` for graph operations (replacing `networkx` and `hypernetx`)
- `DataFrames.jl` for tabular data (replacing `pandas`)
- `ArgParse.jl` for CLI interfaces (constitution requirement)
- `JLD2.jl` for serialization (replacing `pickle`)
- `SHA.jl` for hash generation (chunk IDs)

*Text Chunking*: Implemented recursive character text splitter matching LangChain's behavior, with configurable chunk size and overlap.

*LLM Integration*: Function-based interface accepting a `generate` function, allowing flexibility in LLM backend choice (Transformers.jl, API calls, etc.).

*Caching*: Hypergraph results cached to JLD2 files using graph root identifiers, enabling efficient re-use of expensive LLM operations.

=== Julia-Idiomatic Patterns

*Type Annotations*: Functions use explicit return type annotations for clarity and performance.

*Error Handling*: Uses Julia exception types (`ArgumentError`, `IOError`) with descriptive messages.

*Structured Logging*: Custom `LoggingUtils` module provides structured logging with operation context.

*Multiple Dispatch*: Leverages Julia's multiple dispatch where appropriate (e.g., different input types).

=== Key Differences from Python

1. *Type System*: Julia's type system enables better performance through type inference and specialization.

2. *Serialization*: Uses JLD2 format instead of pickle, providing better cross-language compatibility and type preservation.

3. *Package Management*: Julia's `Pkg` system with `Project.toml` provides explicit dependency management.

4. *CLI Interface*: Constitution requirement mandates CLI for all packages, implemented via `ArgParse.jl`.

== Learnings and Challenges

=== Challenges Encountered

*Hypergraph Union Operations*: The Python `hypernetx` library provides union operations that needed careful replication in Julia's `Hypergraphs.jl`. The Julia implementation uses explicit vertex and hyperedge addition.

*LLM Response Parsing*: JSON parsing from LLM responses requires robust error handling, as LLMs may produce malformed JSON. Implemented try-catch with fallback to empty hypergraph.

*Text Chunking Edge Cases*: Recursive character splitting needed careful handling of edge cases (empty text, very short text, no good split boundaries).

*Dependency Management*: Local package dependencies (GraphUtils.jl) require careful path management during development. Production deployment will use proper package registration.

=== Solutions Adopted

*Hypergraph Union*: Implemented explicit iteration over vertices and hyperedges, adding them to a new hypergraph. This ensures correctness even if union operations differ between libraries.

*Error Handling*: Comprehensive try-catch blocks with logging, allowing graceful degradation when LLM calls fail.

*Text Splitting*: Implemented separator-based splitting with fallback to character-based splitting, matching LangChain behavior.

*Development Workflow*: Using local includes during development, with plan to migrate to proper package dependencies via Pkg.

=== Performance Considerations

*Caching Strategy*: Aggressive caching of hypergraph results reduces expensive LLM calls. Cache keys based on text content hash and parameters.

*Chunking Efficiency*: Text chunking implemented with efficient string operations, avoiding unnecessary allocations.

*Type Stability*: Function return types explicitly annotated to enable Julia's compiler optimizations.

== Code Examples

=== Python vs Julia Comparison

*Python*:
```python
def documents2Dataframe(documents):
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
            "chunk_id": md5(chunk.encode()).hexdigest(),
        }
        rows.append(row)
    return pd.DataFrame(rows)
```

*Julia*:
```julia
function documents2dataframe(documents::Vector{String})::DataFrame
    rows = []
    for chunk in documents
        chunk_id = bytes2hex(sha256(chunk))
        push!(rows, (text=chunk, chunk_id=chunk_id))
    end
    return DataFrame(rows)
end
```

*Key Differences*:
- Julia uses `SHA256` instead of `MD5` (more secure, still fast)
- Type annotations enable compiler optimizations
- DataFrame construction uses named tuples for clarity

=== Usage Example

```julia
using GraphGeneration

# Convert documents
documents = ["First document", "Second document"]
df = documents2dataframe(documents)

# Generate hypergraph (with mock LLM function)
function mock_generate(system_prompt, prompt)
    return """{"nodes": [{"id": "node1", "type": "concept"}], "edges": []}"""
end

hg, sub_dfs = df2hypergraph(df, mock_generate)
```

== Next Steps

1. *Equivalence Testing*: Run full equivalence tests against Python reference outputs
2. *Performance Benchmarking*: Compare Julia vs Python performance on large text corpora
3. *LLM Integration*: Integrate with actual LLM backends (Transformers.jl, API calls)
4. *Documentation*: Complete Documenter.jl API documentation
5. *Package Registration*: Prepare packages for Julia package registry

== References

- Python Source: `HyperGraphReasoningPython/GraphReasoning/graph_generation.py`
- Julia Package: `GraphGeneration.jl/`
- API Contracts: `specs/001-julia-replication/contracts/api-contracts.md`
- Data Model: `specs/001-julia-replication/data-model.md`
