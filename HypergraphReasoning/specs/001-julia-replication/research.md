# Research: Julia Replication Project

**Date**: 2026-01-16  
**Purpose**: Resolve technical unknowns for replicating Python HyperGraphReasoning package in Julia

## Research Questions

### 1. Julia Equivalents for Python Dependencies

**Question**: What Julia packages provide equivalent functionality to Python dependencies?

**Findings**:

#### Core Scientific Computing
- **numpy** → Julia built-in arrays + `LinearAlgebra.jl` (standard library)
- **pandas** → `DataFrames.jl` (primary), `DataFramesMeta.jl` (syntax sugar)
- **scipy** → `SciPy.jl` (wrapper) or native Julia: `LinearAlgebra.jl`, `Statistics.jl`, `Distributions.jl`, `Optim.jl`
- **matplotlib** → `Plots.jl` (primary, backend-agnostic), `PyPlot.jl` (matplotlib wrapper), `Makie.jl` (modern, GPU-accelerated)
- **seaborn** → `StatsPlots.jl` (statistical plotting), `AlgebraOfGraphics.jl`

#### Graph/Hypergraph Libraries
- **networkx** → `Graphs.jl` (primary, actively maintained), `LightGraphs.jl` (legacy, being phased out), `MetaGraphs.jl` (graph with metadata)
- **hypernetx** → `Hypergraphs.jl` (primary Julia hypergraph library), `SimpleHypergraphs.jl` (simpler alternative)
- **python-louvain** → `CommunityDetection.jl` (includes Louvain algorithm)

#### Machine Learning / LLM
- **scikit-learn** → `MLJ.jl` (comprehensive ML framework), `ScikitLearn.jl` (wrapper), `Flux.jl` (deep learning)
- **transformers** → `Transformers.jl` (HuggingFace transformers port), `TextModels.jl`
- **sentence-transformers** → `Embeddings.jl` or use `Transformers.jl` with sentence transformer models
- **langchain** → No direct equivalent; implement custom agent framework or use `PromptingTools.jl` for LLM interactions
- **openai** → `OpenAI.jl` (OpenAI API client)

#### Utilities
- **tqdm** → `ProgressMeter.jl` (progress bars)
- **pickle** → `Serialization.jl` (standard library), `JLD2.jl` (HDF5-based, recommended)
- **hashlib.md5** → `SHA.jl` (standard library, includes MD5)
- **json** → `JSON.jl` (standard library)

#### Text Processing
- **langchain.text_splitter** → Custom implementation or `TextAnalysis.jl` for text processing
- **pdfminer.six** → `PDFIO.jl` or `Poppler.jl` for PDF parsing
- **markdown2** → `Markdown.jl` (standard library)

**Decision**: Use `Graphs.jl` for graph operations, `Hypergraphs.jl` for hypergraph operations, `DataFrames.jl` for tabular data, `Plots.jl` for visualization, `MLJ.jl` for ML, `Transformers.jl` for LLM integration, `JLD2.jl` for serialization.

**Rationale**: These are mature, actively maintained Julia packages that align with Julia ecosystem best practices. `Graphs.jl` is the recommended successor to `LightGraphs.jl`. `Hypergraphs.jl` provides native Julia hypergraph support.

**Alternatives Considered**: 
- `LightGraphs.jl` (being phased out in favor of `Graphs.jl`)
- `PyCall.jl` to call Python libraries directly (rejected - defeats purpose of native Julia implementation)
- `SimpleHypergraphs.jl` (simpler but less feature-complete than `Hypergraphs.jl`)

---

### 2. CLI Interface Requirements

**Question**: How should Julia packages expose CLI functionality per Constitution Principle II?

**Findings**:

Julia packages typically expose CLI via:
1. **CommandLine.jl** - Simple CLI argument parsing
2. **ArgParse.jl** - More advanced argument parsing (similar to Python argparse)
3. **Julia CLI scripts** - Executable scripts using `julia --project` or `julia -e`
4. **PackageCommand.jl** - Command registration system

For text I/O (stdin/stdout/stderr):
- Julia supports standard streams: `stdin`, `stdout`, `stderr`
- `JSON.jl` for JSON format support
- `Printf.jl` for formatted output

**Decision**: Each Julia package will include CLI tools using `ArgParse.jl` for argument parsing. CLI commands will:
- Accept input via stdin or file arguments
- Output to stdout (JSON or human-readable format)
- Write errors to stderr
- Be executable via `julia --project=<package> -e 'using <Package>; <Package>.cli_main()'` or standalone scripts

**Rationale**: `ArgParse.jl` provides robust CLI argument handling similar to Python's argparse, which the Python package may use. Text I/O protocol enables Unix tool composability and debugging.

**Alternatives Considered**:
- No CLI (rejected - violates Constitution Principle II)
- `CommandLine.jl` (simpler but less feature-rich than `ArgParse.jl`)
- Python-style CLI scripts (Julia-native approach preferred)

---

### 3. Julia Package Structure Best Practices

**Question**: What is the standard structure for Julia packages?

**Findings**:

Standard Julia package structure:
```
PackageName.jl/
├── src/
│   └── PackageName.jl      # Main module file
├── test/
│   └── runtests.jl         # Test suite
├── docs/
│   ├── src/                # Documenter.jl source
│   └── make.jl             # Documenter build script
├── Project.toml            # Dependencies and metadata
└── README.md
```

Package naming: PascalCase (e.g., `GraphGeneration.jl`)

**Decision**: Follow standard Julia package structure. Each package will have:
- `src/<PackageName>.jl` - Main module
- `test/runtests.jl` - Test suite including equivalence tests
- `docs/src/` - Documenter.jl documentation
- `Project.toml` - Dependencies (may include other top-level Julia packages)
- CLI scripts in `bin/` or as exported functions callable via `-e`

**Rationale**: Standard structure ensures compatibility with Julia package manager (Pkg.jl), enables proper dependency resolution, and follows ecosystem conventions.

---

### 4. Testing Strategy for Equivalence Validation

**Question**: How to implement automated tests comparing Julia and Python outputs?

**Findings**:

Options for equivalence testing:
1. **Test.jl with external Python calls** - Use `PyCall.jl` or subprocess to run Python code and compare outputs
2. **Reference data files** - Generate reference outputs from Python, store as test fixtures, compare Julia outputs
3. **Property-based testing** - Test mathematical properties rather than exact output matching
4. **Tolerance-based comparison** - Use approximate equality for floating-point comparisons

For hypergraph data structures:
- Serialize both Python and Julia hypergraphs to JSON or JLD2
- Compare serialized representations
- Use tolerance for floating-point embeddings

**Decision**: Use hybrid approach:
- **Automated tests**: Generate reference outputs from Python (stored as test fixtures), compare Julia outputs using `Test.jl` with tolerance for floating-point values
- **Manual verification**: For complex cases, visual inspection and manual validation
- **Test fixtures**: Store Python reference outputs as JSON/JLD2 files in `test/fixtures/`
- **Comparison utilities**: Create test helper functions for comparing hypergraphs, embeddings, and analysis results

**Rationale**: Reference data approach avoids requiring Python runtime in Julia test environment while enabling systematic comparison. Tolerance-based comparison handles floating-point precision differences.

**Alternatives Considered**:
- `PyCall.jl` to call Python directly (adds Python dependency, rejected)
- Exact equality (too strict for floating-point, rejected)
- Property-based only (may miss edge cases, rejected as sole approach)

---

### 5. Documentation Approach (Documenter.jl + Typst)

**Question**: How to structure dual documentation system (Documenter.jl + Typst)?

**Findings**:

**Documenter.jl**:
- Standard Julia package documentation tool
- Generates HTML documentation from Markdown
- Integrates with Julia docstrings
- Builds via `julia docs/make.jl`

**Typst**:
- Modern typesetting system (alternative to LaTeX)
- Compiles `.typ` files to PDF
- Better than Markdown for formal documentation
- Can include code blocks, math, figures

**Decision**: 
- **Documenter.jl**: Each package includes `docs/src/` with API documentation. Focus on function signatures, examples, API reference.
- **Typst**: Global documentation in `documentation/` directory covering:
  - `progress.typ` - Replication progress tracking
  - `implementation.typ` - Design decisions, implementation approaches
  - `learnings.typ` - Knowledge acquired, challenges, solutions
- Typst docs can reference package Documenter.jl docs via links

**Rationale**: Separation of concerns - API docs (Documenter.jl) vs. project documentation (Typst). Documenter.jl integrates with Julia ecosystem. Typst provides professional typesetting for research-style documentation.

---

## Resolved Technical Context

Based on research findings, updated technical context:

**Language/Version**: Julia 1.9+ (latest stable)  
**Primary Dependencies**: 
- `Graphs.jl` (graph operations)
- `Hypergraphs.jl` (hypergraph operations)
- `DataFrames.jl` (tabular data)
- `Plots.jl` (visualization)
- `MLJ.jl` (machine learning)
- `Transformers.jl` (LLM integration)
- `ArgParse.jl` (CLI argument parsing)
- `JLD2.jl` (serialization)
- `JSON.jl` (JSON I/O)
- `Documenter.jl` (API documentation)
- `Test.jl` (testing framework)

**Storage**: Files (JLD2/JSON/CSV for hypergraph data, GraphML for graph export, text files for documentation)  
**Testing**: `Test.jl` with reference data fixtures + manual verification  
**Target Platform**: Cross-platform (Julia)  
**Project Type**: Multiple Julia packages  
**Performance Goals**: Match or exceed Python performance  
**Constraints**: Functional equivalence, Julia-idiomatic code, CLI interface per package, independent installability

---

## Constitution Check Update (Post-Research)

### II. CLI Interface
✅ **PASS**: Each Julia package will expose CLI functionality using `ArgParse.jl`. Text I/O protocol: stdin/args → stdout, errors → stderr. JSON and human-readable formats supported. Executable via `julia --project` or standalone scripts.

**Gate Status**: ✅ **PASS** - All constitution requirements satisfied.

---

### 6. Notebook Conversion Approach

**Question**: How should Python Jupyter notebooks be converted to Julia format?

**Findings**:

Julia notebook options:
1. **Pluto.jl** - Reactive notebooks (`.jl` files), native Julia, interactive execution
2. **Jupyter.jl** - Julia kernel for Jupyter notebooks (keeps `.ipynb` format)
3. **Weave.jl** - Literate programming (`.jmd` files), documentation-focused
4. **Franklin.jl** - Static site generator with notebook support

**Pluto.jl characteristics**:
- Native Julia reactive notebook format
- `.jl` files (plain Julia code with special cell markers)
- Automatic dependency management via `using` statements
- Reactive execution (cells update when dependencies change)
- No hidden state (reproducible)
- Can be version-controlled as plain text

**Conversion approach**:
- Manual conversion: Translate Python code to Julia, replace Python imports with Julia `using` statements
- Automated tools: Limited - most conversion is manual due to language differences
- Structure preservation: Maintain notebook cell structure and workflow
- Output preservation: Convert outputs where possible (data, visualizations)

**Decision**: Convert Python Jupyter notebooks to Pluto notebooks (`.jl` files). Conversion process:
1. Translate Python code cells to Julia equivalents
2. Replace Python imports with Julia `using` statements (using the top-level Julia packages)
3. Convert Python-specific syntax to Julia (e.g., list comprehensions, dict operations)
4. Maintain notebook structure and workflow
5. Store in `notebooks/` directory at repository root, mirroring Python Notebooks/ structure
6. Validate via automated output comparison (where possible) and manual execution verification

**Rationale**: Pluto is the native Julia notebook format, provides reactive execution, and aligns with Julia best practices. Plain text format (`.jl` files) enables version control and reproducibility. Dependencies are managed within notebook files via `using` statements.

**Alternatives Considered**:
- Jupyter.jl (keeps `.ipynb` format, but less Julia-native)
- Weave.jl (`.jmd` files, more documentation-focused, less interactive)
- Manual conversion to scripts (loses notebook interactivity)

---

## Resolved Technical Context

Based on research findings, updated technical context:

**Language/Version**: Julia 1.9+ (latest stable)  
**Primary Dependencies**: 
- `Graphs.jl` (graph operations)
- `Hypergraphs.jl` (hypergraph operations)
- `DataFrames.jl` (tabular data)
- `Plots.jl` (visualization)
- `MLJ.jl` (machine learning)
- `Transformers.jl` (LLM integration)
- `ArgParse.jl` (CLI argument parsing)
- `JLD2.jl` (serialization)
- `JSON.jl` (JSON I/O)
- `Documenter.jl` (API documentation)
- `Test.jl` (testing framework)
- `Pluto.jl` (reactive notebooks for conversion and execution)

**Storage**: Files (JLD2/JSON/CSV for hypergraph data, GraphML for graph export, text files for documentation, Pluto notebooks as `.jl` files)  
**Testing**: `Test.jl` with reference data fixtures + manual verification + notebook execution validation  
**Target Platform**: Cross-platform (Julia)  
**Project Type**: Multiple Julia packages + Pluto notebooks  
**Performance Goals**: Match or exceed Python performance  
**Constraints**: Functional equivalence, Julia-idiomatic code, CLI interface per package, independent installability, notebook conversion with equivalent functionality

---

## Constitution Check Update (Post-Research)

### II. CLI Interface
✅ **PASS**: Each Julia package will expose CLI functionality using `ArgParse.jl`. Text I/O protocol: stdin/args → stdout, errors → stderr. JSON and human-readable formats supported. Executable via `julia --project` or standalone scripts. Note: Pluto notebooks are interactive environments and do not require CLI interfaces (executed via Pluto.jl).

**Gate Status**: ✅ **PASS** - All constitution requirements satisfied.
