# Implementation Plan: Julia Replication Project

**Branch**: `001-julia-replication` | **Date**: 2026-01-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/docs/hypergraph-reasoning/specs/001-julia-replication/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Replicate the Python HyperGraphReasoning package in Julia by creating one Julia package per Python module as a top-level `*.jl/` directory at repository root. Each package will maintain functional equivalence (same logical behavior and outputs) while using Julia-idiomatic implementations optimized for Julia performance. Documentation will use Documenter.jl for API docs within packages and Typst for global progress/learnings documentation. Development standards will be maintained in AGENTS.md. Additionally, convert all Python Jupyter notebooks to Pluto notebooks stored in @notebooks directory, maintaining equivalent workflows using those top-level Julia packages.

**Technical Approach**: 
- Use `Graphs.jl` and `Hypergraphs.jl` for graph/hypergraph operations
- Use `DataFrames.jl` for tabular data processing
- Use `Transformers.jl` for LLM integration
- Use `ArgParse.jl` for CLI interfaces (constitution requirement)
- Use `JLD2.jl` for serialization
- Use `Pluto.jl` for notebook conversion and execution
- Implement TDD with automated equivalence tests comparing Julia/Python outputs
- Convert Python Jupyter notebooks to Pluto notebooks with equivalent functionality

## Technical Context

**Language/Version**: Julia 1.9+ (latest stable)  
**Primary Dependencies**: 
- `Graphs.jl` (graph operations, replaces networkx)
- `Hypergraphs.jl` (hypergraph operations, replaces hypernetx)
- `DataFrames.jl` (tabular data, replaces pandas)
- `Plots.jl` (visualization, replaces matplotlib/seaborn)
- `MLJ.jl` (machine learning, replaces scikit-learn)
- `Transformers.jl` (LLM integration, replaces transformers)
- `ArgParse.jl` (CLI argument parsing)
- `JLD2.jl` (serialization, replaces pickle)
- `JSON.jl` (JSON I/O, standard library)
- `Documenter.jl` (API documentation)
- `Test.jl` (testing framework, standard library)
- `Pluto.jl` (reactive notebooks, for notebook conversion and execution)
- `SciPy.jl` or native Julia scientific computing libraries (replaces scipy)  
**Storage**: Files (JLD2/CSV/GraphML formats for hypergraph data, text files for documentation, Pluto notebooks as `.jl` files)  
**Testing**: Test.jl (Julia standard testing framework) + automated equivalence tests comparing Julia/Python outputs  
**Target Platform**: Cross-platform (Julia supports Linux, macOS, Windows)  
**Project Type**: Multiple Julia packages (one package per Python module: GraphGeneration.jl, GraphAnalysis.jl, GraphTools.jl, GraphUtils.jl) + Pluto notebooks in @notebooks directory  
**Performance Goals**: Match or exceed Python implementation performance; leverage Julia's JIT compilation for scientific computing workloads  
**Constraints**: Must maintain functional equivalence with Python implementation; Julia-idiomatic code required; packages must be independently installable; converted notebooks must execute successfully and produce equivalent results  
**Scale/Scope**: Scientific computing workloads (hypergraph generation from text corpora, analysis of large hypergraphs). Note: Agentic reasoning capabilities are deferred to future phases beyond MVP. Notebook conversion includes 3 notebooks: make_hypergraph.ipynb, Analyze_Hypergraph.ipynb, and Agents.ipynb.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Library-First
✅ **PASS**: Each Julia package is a standalone library (one per Python module). Packages are self-contained, independently testable, and documented. Clear purpose: replicate specific Python module functionality. Pluto notebooks are separate artifacts that use these packages.

### II. CLI Interface
✅ **PASS**: Each Julia package will expose CLI functionality using `ArgParse.jl`. CLI interfaces are defined in contracts for all packages. Tasks implement CLI for GraphGeneration.jl and GraphUtils.jl in User Story 1. CLI for GraphAnalysis.jl and GraphTools.jl will be added when those packages are implemented in future phases. Note: Pluto notebooks are interactive environments and do not require CLI interfaces (they are executed via Pluto.jl).

### III. Test-First (NON-NEGOTIABLE)
✅ **PASS**: Test-Driven Development mandatory. Automated equivalence tests plus manual verification specified. Tests written before implementation per TDD cycle. Notebook conversion validation includes both automated output comparison and manual verification.

### IV. Integration Testing
✅ **PASS**: Integration tests required for: contract tests (Julia package functionality), inter-package communication (package dependencies), shared schemas (data format compatibility), notebook execution (converted notebooks using packages). End-to-end workflows must be verified.

### V. Observability, Versioning & Simplicity
✅ **PASS**: Structured logging required. Semantic versioning (MAJOR.MINOR.PATCH) for packages. Start simple (YAGNI). Complexity must be justified in documentation.

**Gate Status (Pre-Research)**: ⚠️ **CONDITIONAL PASS** - CLI interface requirement needed clarification.

**Gate Status (Post-Research)**: ✅ **PASS** - All constitution requirements satisfied. CLI interface will use `ArgParse.jl` per research findings.

**Gate Status (Post-Design)**: ✅ **PASS** - All constitution requirements satisfied. CLI interfaces defined in contracts for all packages. CLI implementation tasks included for GraphGeneration.jl and GraphUtils.jl in User Story 1. CLI for remaining packages (GraphAnalysis.jl, GraphTools.jl) will be added when those packages are implemented. TDD approach established, integration testing strategy defined. Notebook conversion approach defined (Pluto notebooks, validation strategy).

## Project Structure

### Documentation (this feature)

```text
docs/hypergraph-reasoning/specs/001-julia-replication/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

**Structure Decision**: Multiple Julia packages structure. Each Python module maps to one top-level Julia package directory. Pluto notebooks stored in @notebooks directory:

```text
GraphGeneration.jl/          # Replicates graph_generation.py
├── src/
│   └── GraphGeneration.jl
├── test/
├── docs/
│   └── src/                 # Documenter.jl docs
└── Project.toml
GraphAnalysis.jl/            # Replicates graph_analysis.py
├── src/
│   └── GraphAnalysis.jl
├── test/
├── docs/
└── Project.toml
GraphTools.jl/               # Replicates graph_tools.py
├── src/
│   └── GraphTools.jl
├── test/
├── docs/
└── Project.toml
GraphUtils.jl/               # Replicates utils.py
├── src/
│   └── GraphUtils.jl
├── test/
├── docs/
└── Project.toml

notebooks/                    # Converted Pluto notebooks
├── SG/                       # Mirrors Python Notebooks/SG/ structure
│   ├── make_hypergraph.jl   # Converted from make_hypergraph.ipynb
│   ├── Analyze_Hypergraph.jl # Converted from Analyze_Hypergraph.ipynb
│   └── Agents.jl             # Converted from Agents.ipynb
│   └── [subdirectories mirroring Python structure as needed]

docs/hypergraph-reasoning/    # Integrated HypergraphReasoning documentation
├── graph-generation.typ
├── specs/
│   └── 001-julia-replication/
└── test/

AGENTS.md                     # Development standards
```

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations - all constitution requirements satisfied.

**Justification**: 
- Multiple packages (4) are necessary to mirror Python module structure and enable independent development/testing
- Package dependencies are minimal and justified by Python import relationships
- CLI interfaces are required by constitution and implemented via ArgParse.jl
- Notebook conversion adds complexity but is required for complete replication; Pluto notebooks are standard Julia practice

## Generated Artifacts

### Phase 0: Research
- ✅ `research.md` - Resolved all technical unknowns (Julia package equivalents, CLI approach, testing strategy, notebook conversion approach)

### Phase 1: Design
- ✅ `data-model.md` - Data structures and entities (Hypergraph, Node Embeddings, Text Chunk, Pluto Notebook, etc.)
- ✅ `contracts/api-contracts.md` - Function signatures and CLI interfaces for all packages
- ✅ `quickstart.md` - Development workflow and setup instructions
- ✅ Agent context updated (`.cursor/rules/specify-rules.mdc`)

## Next Steps

1. Run `/speckit.tasks` to generate task breakdown (including notebook conversion tasks)
2. Begin implementation with User Story 1 (P1): Replicate core functionality
3. Follow TDD: Write tests → Tests fail → Implement → Tests pass
4. Convert Python notebooks to Pluto notebooks
5. Update documentation as implementation progresses
