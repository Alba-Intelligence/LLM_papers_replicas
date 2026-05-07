# HypergraphReasoning documentation

The repository now keeps the remaining HypergraphReasoning planning and narrative material under `docs/hypergraph-reasoning/` instead of leaving it in the old `HypergraphReasoning/` subtree.

## What lives there

- `docs/hypergraph-reasoning/graph-generation.typ` - the Typst progress and learnings document for the Julia `GraphGeneration.jl` replication.
- `docs/hypergraph-reasoning/test/verify_typst.jl` - the lightweight Typst verification script.
- `docs/hypergraph-reasoning/specs/001-julia-replication/` - the original specification bundle, including the feature spec, research notes, design docs, contracts, quickstart, and task breakdown.

## Why it is organized this way

The Julia packages and shared notebooks were already moved to repository-level homes, so keeping the associated HypergraphReasoning documentation under the root `docs/` tree makes the repository easier to navigate. It also keeps the narrative wiki, the shared Documenter site, and the preserved HypergraphReasoning design record in one documentation namespace.

## Suggested reading order

1. `docs/hypergraph-reasoning/graph-generation.typ`
2. `docs/hypergraph-reasoning/specs/001-julia-replication/spec.md`
3. `docs/hypergraph-reasoning/specs/001-julia-replication/plan.md`
4. `docs/hypergraph-reasoning/specs/001-julia-replication/contracts/api-contracts.md`
5. `docs/hypergraph-reasoning/specs/001-julia-replication/tasks.md`
