# Feature Specification: Julia Replication Project

**Feature Branch**: `001-julia-replication`  
**Created**: 2026-01-16  
**Status**: Draft  
**Input**: User description: "The project is several fold: (1) to replicate a Python package located in @HyperGraphReasoning in the Julia programming language. The Julia packages will be created in @replica. (2) Maintain a full set of documentation of your progress and acquired knowledge in @documentation. The documentation will be formatted in Typst (not traditional Markdown) (3) Maintain the @AGENTS.md to ensure taht any work maintains strong standards and sold starting points. Do you have any questions?"

## Clarifications

### Session 2026-01-16

- Q: How should Julia packages be structured relative to Python modules? → A: One Julia package per Python module (Option B), with possibility to refactor later
- Q: What does "feature parity" mean for validation? → A: Functional equivalence with Julia-idiomatic design (Option C) - same logical behavior, optimized for Julia
- Q: How should documentation be structured? → A: Mixed structure (Option D) - Documenter.jl in each Julia package for API documentation, plus global Typst documentation for progress, learnings, and design decisions
- Q: Can Julia packages depend on each other? → A: Yes (Option B) - Julia packages can depend on each other with explicit Project.toml dependencies, mirroring Python module imports
- Q: How should equivalence be validated? → A: Automated tests for output comparison plus manual verification for complex/edge cases (Option C)

### Session 2026-01-16 (Notebook Conversion)

- Q: What format should be used for converting Python Jupyter notebooks? → A: Pluto notebooks (Option B) - Convert to native Julia reactive notebooks (.jl files) for interactive execution
- Q: Which notebooks should be converted in MVP scope? → A: All notebooks (Option A) - Convert all 3 notebooks (make_hypergraph, Analyze_Hypergraph, Agents) in MVP scope
- Q: Where should converted Pluto notebooks be stored? → A: notebooks/ directory at repository root (Option A) - Mirrors Python structure, separate from packages
- Q: How should notebook dependencies be managed? → A: Dependencies specified within notebook files via `using` statements (Pluto's native approach) - Pluto automatically manages the environment
- Q: How should converted notebook equivalence be validated? → A: Both automated and manual (Option C) - Automated output comparison where possible, plus manual verification for interactive/visual outputs

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Replicate Python Package Core Functionality (Priority: P1)

A developer needs to use the hypergraph reasoning capabilities from the Python package in Julia. They should be able to install and use Julia packages that provide equivalent functionality to the Python GraphReasoning package, including hypergraph generation and analysis capabilities. Additionally, they should be able to use converted Pluto notebooks that replicate the workflows from Python Jupyter notebooks. (Note: Agentic reasoning tools are deferred to future phases beyond MVP scope.)

**Why this priority**: This is the core deliverable - without functional Julia packages, the project cannot provide value. The replication must maintain feature parity with the Python implementation.

**Independent Test**: Can be fully tested by creating a Julia package in the @replica directory that implements at least one core function from the Python package (e.g., hypergraph generation from text) and verifying it produces equivalent results to the Python version. Additionally, converted Pluto notebooks in @notebooks should execute successfully and produce equivalent outputs to the original Python notebooks.

**Acceptance Scenarios**:

1. **Given** the Python package structure in HyperGraphReasoningPython/, **When** a developer creates a Julia package in @replica, **Then** the package provides equivalent functionality to at least one core module (graph_generation, graph_analysis, or graph_tools)
2. **Given** a Julia package in @replica, **When** a developer uses it to generate a hypergraph from text, **Then** the output matches the Python implementation's behavior and data structures
3. **Given** multiple Julia packages in @replica, **When** a developer installs and uses them, **Then** they work independently and can be composed together

---

### User Story 2 - Maintain Typst Documentation (Priority: P2)

A developer or researcher needs to understand the replication process, design decisions, and knowledge acquired during the project. They should be able to access comprehensive documentation in Typst format that captures progress, implementation details, and learnings.

**Why this priority**: Documentation enables knowledge transfer, maintains project continuity, and helps future developers understand design decisions. Typst format ensures professional, typeset documentation.

**Independent Test**: Can be fully tested by creating at least one Typst document in @documentation that describes the replication of one Python module, including implementation approach, challenges encountered, and solutions adopted.

**Acceptance Scenarios**:

1. **Given** a Julia package module is replicated, **When** documentation is created in @documentation, **Then** it is formatted in Typst and includes progress notes, implementation details, and acquired knowledge
2. **Given** documentation exists in @documentation, **When** a developer reads it, **Then** they can understand the replication approach, design decisions, and any challenges encountered
3. **Given** multiple documentation files in @documentation, **When** they are compiled, **Then** they produce a cohesive, well-formatted document set

---

### User Story 3 - Maintain Development Standards via AGENTS.md (Priority: P3)

A developer needs clear guidance on development standards and starting points to ensure consistent, high-quality work across the project. They should be able to reference AGENTS.md to understand project conventions, coding standards, and best practices.

**Why this priority**: Maintaining standards ensures code quality, consistency, and reduces onboarding time. While less critical than core functionality, it significantly impacts long-term maintainability.

**Independent Test**: Can be fully tested by ensuring AGENTS.md exists, contains relevant development standards, and is updated when new conventions or practices are established during the replication process.

**Acceptance Scenarios**:

1. **Given** AGENTS.md exists, **When** a developer references it, **Then** they find clear guidance on development standards, coding conventions, and project best practices
2. **Given** new development patterns emerge during replication, **When** they are documented, **Then** AGENTS.md is updated to reflect these standards
3. **Given** AGENTS.md contains standards, **When** code is written, **Then** it adheres to the documented conventions

---

### Edge Cases

- What happens when the Python package structure changes during replication? The Julia packages should be designed to handle version differences and document any deviations.
- How does the system handle missing or incomplete Python source code? Documentation should note gaps and alternative approaches taken.
- What if Typst compilation fails? Documentation should be maintainable in source form even if compilation is temporarily unavailable.
- How are breaking changes in Python dependencies handled? Julia packages should document compatibility requirements and migration paths.
- How are test failures in automated equivalence tests handled? Automated tests compare outputs programmatically; failures trigger manual verification to determine if differences are acceptable (Julia-idiomatic optimizations) or represent actual bugs.
- How are Python Jupyter notebook outputs and execution state handled during conversion? Converted Pluto notebooks should replicate the workflow and produce equivalent results, but execution state from Python notebooks is not preserved. Notebooks are converted to use Julia packages from @replica. Validation uses both automated output comparison (where outputs can be extracted programmatically) and manual verification (for interactive/visual outputs).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST create Julia packages in the @replica directory that replicate functionality from HyperGraphReasoningPython/GraphReasoning/. For MVP scope, at least one Julia package (GraphGeneration.jl) MUST be fully implemented. Each Python module (graph_generation, graph_analysis, graph_tools, utils) maps to one corresponding Julia package (e.g., GraphGeneration.jl, GraphAnalysis.jl). Remaining packages (GraphAnalysis.jl, GraphTools.jl) may be implemented in future phases. Refactoring to different structure may occur later.
- **FR-010**: System MUST convert all Python Jupyter notebooks (`.ipynb` files) from HyperGraphReasoningPython/Notebooks/SG/ to Pluto notebooks (`.jl` files) in MVP scope. This includes: make_hypergraph.ipynb, Analyze_Hypergraph.ipynb, and Agents.ipynb. Converted notebooks MUST be stored in `notebooks/` directory at repository root, maintaining directory structure mirroring the Python Notebooks/ organization. Converted notebooks MUST maintain equivalent functionality and workflow, using Julia packages from @replica instead of Python modules.
- **FR-002**: System MUST maintain functional equivalence with Python implementation for core hypergraph operations (generation, analysis, tools). Equivalence means same logical behavior and outputs, but implementation should be Julia-idiomatic and optimized for Julia performance patterns.
- **FR-003**: System MUST create and maintain documentation using two approaches: (1) Documenter.jl for API documentation within each Julia package (standard Julia practice), (2) Typst format files in @documentation directory for progress, implementation details, design decisions, and acquired knowledge. Each Julia package MUST include Documenter.jl-generated API documentation. Typst files MUST document progress, implementation details, design decisions, and acquired knowledge for replicated modules.
- **FR-005**: System MUST maintain AGENTS.md file with development standards, coding conventions, and project best practices
- **FR-006**: System MUST update AGENTS.md when new development patterns or standards are established
- **FR-007**: Julia packages MUST be independently installable and usable. Packages MAY declare dependencies on other Julia packages in @replica via explicit Project.toml entries, mirroring Python module import relationships.
- **FR-008**: Typst documentation MUST be compilable to produce formatted output (PDF or other Typst output formats). Documenter.jl documentation MUST be buildable to HTML or other supported formats.
- **FR-009**: System MUST preserve the directory structure reference system (@replica, @documentation, @HyperGraphReasoning, @notebooks)

### Key Entities *(include if feature involves data)*

- **Julia Package**: A Julia package in @replica directory that replicates one Python module's functionality. Structure: one Julia package per Python module (e.g., GraphGeneration.jl for graph_generation.py). Key attributes: package name (maps to Python module name), module structure, exported functions, dependencies (may include other @replica packages via Project.toml). May be refactored to different organization later.
- **Pluto Notebook**: A converted Julia notebook (`.jl` file) that replicates functionality from a Python Jupyter notebook (`.ipynb` file). Key attributes: notebook name (maps to original Python notebook), reactive cells, Julia package dependencies specified via `using` statements within the notebook (Pluto manages environment automatically), equivalent workflow and outputs. Stored in `notebooks/` directory at repository root, maintaining directory structure mirroring the Python Notebooks/ organization.
- **Documentation File**: Two types: (1) Typst-formatted documents in @documentation directory covering progress, implementation details, design decisions, and learnings. Key attributes: file name, content sections, compilation status. (2) Documenter.jl API documentation within each Julia package (standard Julia package documentation).
- **Development Standards**: Guidelines and conventions stored in AGENTS.md. Key attributes: coding standards, project structure, best practices, update history.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: At least one Julia package in @replica successfully replicates core functionality from one Python module (graph_generation, graph_analysis, or graph_tools) with functional equivalence (same logical behavior and outputs, Julia-idiomatic implementation). Validation via automated tests comparing outputs plus manual verification for complex cases.
- **SC-002**: Documentation covers at least one replicated module: (1) Typst documentation in @documentation with progress notes, implementation approach, and acquired knowledge, (2) Documenter.jl API documentation in the corresponding Julia package
- **SC-003**: AGENTS.md contains at least 5 distinct development standards or conventions that guide Julia package development
- **SC-004**: Julia packages can be installed and used independently by developers without requiring the Python source code. Package dependencies are resolved via Project.toml declarations.
- **SC-005**: Typst documentation files compile successfully to produce formatted output (PDF or equivalent). Documenter.jl documentation builds successfully to HTML or other supported formats.
- **SC-006**: All three project components (@replica packages, @documentation, AGENTS.md) are maintained and updated as replication progresses
- **SC-007**: All Python Jupyter notebooks from HyperGraphReasoningPython/Notebooks/SG/ are successfully converted to Pluto notebooks in @notebooks directory. Converted notebooks execute successfully, use Julia packages from @replica, and produce equivalent results to Python versions. Validation via automated output comparison where possible, plus manual verification for interactive/visual outputs.
