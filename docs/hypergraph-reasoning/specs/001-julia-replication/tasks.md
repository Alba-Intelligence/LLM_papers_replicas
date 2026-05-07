---
description: "Task list for Julia Replication Project"
---

# Tasks: Julia Replication Project

**Input**: Design documents from `/docs/hypergraph-reasoning/specs/001-julia-replication/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are REQUIRED per Constitution Principle III (Test-First, NON-NEGOTIABLE). All tests must be written first and fail before implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Multiple Julia packages**: top-level `<PackageName>.jl/` directories at repository root
- **Documentation**: `docs/hypergraph-reasoning/` for Typst and migrated design records, `<PackageName>.jl/docs/` for Documenter.jl
- **Tests**: `<PackageName>.jl/test/` per package
- **Notebooks**: `notebooks/` at repository root (Pluto notebooks converted from Python Jupyter notebooks)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create top-level Julia package directory structure at repository root
- [X] T002 [P] Create documentation/ directory for Typst files at repository root
- [X] T003 [P] Initialize GraphGeneration.jl package in GraphGeneration.jl using Pkg.generate()
- [X] T004 [P] Initialize GraphAnalysis.jl package in GraphAnalysis.jl using Pkg.generate()
- [X] T005 [P] Initialize GraphTools.jl package in GraphTools.jl using Pkg.generate()
- [X] T006 [P] Initialize GraphUtils.jl package in GraphUtils.jl using Pkg.generate()

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T007 Create test fixtures directory structure: GraphGeneration.jl/test/fixtures/ for Python reference outputs
- [X] T008 [P] Add core dependencies to GraphGeneration.jl/Project.toml: Graphs, Hypergraphs, DataFrames, ArgParse, JSON, JLD2, SHA, Test
- [X] T009 [P] Add core dependencies to GraphAnalysis.jl/Project.toml: Graphs, Hypergraphs, DataFrames, ArgParse, JSON, JLD2, Test, CommunityDetection
- [X] T010 [P] Add core dependencies to GraphTools.jl/Project.toml: Graphs, Hypergraphs, DataFrames, ArgParse, JSON, JLD2, Transformers, Test, LinearAlgebra
- [X] T011 [P] Add core dependencies to GraphUtils.jl/Project.toml: DataFrames, ArgParse, JSON, Test
- [X] T012 [P] Create test helper module for equivalence testing in GraphGeneration.jl/test/test_helpers.jl (comparison utilities for hypergraphs, embeddings)
- [X] T013 [P] Create test helper module for equivalence testing in GraphAnalysis.jl/test/test_helpers.jl
- [X] T014 [P] Create test helper module for equivalence testing in GraphTools.jl/test/test_helpers.jl
- [X] T015 Setup structured logging infrastructure (create logging utility module shared across packages)
- [X] T016 Create Python reference data generation script in test/fixtures/generate_references.py to produce test fixtures

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Replicate Python Package Core Functionality (Priority: P1) 🎯 MVP

**Goal**: Create at least one Julia package (GraphGeneration.jl) that replicates core functionality from graph_generation.py with functional equivalence

**Independent Test**: Create GraphGeneration.jl package, implement documents2dataframe function, run equivalence test comparing Julia output to Python reference data. Test passes when outputs match within tolerance.

### Tests for User Story 1 (REQUIRED - TDD mandatory)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T017 [P] [US1] Write equivalence test for documents2dataframe in GraphGeneration.jl/test/test_documents2dataframe.jl
- [X] T018 [P] [US1] Write equivalence test for df2hypergraph in GraphGeneration.jl/test/test_df2hypergraph.jl
- [X] T019 [P] [US1] Write equivalence test for make_hypergraph_from_text in GraphGeneration.jl/test/test_make_hypergraph.jl
- [X] T020 [P] [US1] Write integration test for end-to-end hypergraph generation workflow in GraphGeneration.jl/test/test_integration.jl
- [X] T021 [P] [US1] Write CLI interface test in GraphGeneration.jl/test/test_cli.jl

### Implementation for User Story 1

- [X] T022 [P] [US1] Create GraphUtils.jl module structure in GraphUtils.jl/src/GraphUtils.jl (extract, remove_markdown_symbols functions)
- [X] T023 [US1] Implement extract function in GraphUtils.jl/src/GraphUtils.jl
- [X] T024 [US1] Implement remove_markdown_symbols function in GraphUtils.jl/src/GraphUtils.jl
- [X] T025 [P] [US1] Write CLI interface test for GraphUtils.jl in GraphUtils.jl/test/test_cli.jl
- [X] T026 [US1] Implement CLI interface using ArgParse.jl in GraphUtils.jl/src/GraphUtils.jl (cli_main function)
- [X] T027 [US1] Add GraphUtils.jl as dependency to GraphGeneration.jl/Project.toml
- [X] T028 [US1] Implement documents2dataframe function in GraphGeneration.jl/src/GraphGeneration.jl
- [X] T029 [US1] Implement text chunking utility (RecursiveCharacterTextSplitter equivalent) in GraphGeneration.jl/src/GraphGeneration.jl
- [X] T030 [US1] Implement hypergraphPrompt function (LLM-based hypergraph extraction) in GraphGeneration.jl/src/GraphGeneration.jl
- [X] T031 [US1] Implement df2hypergraph function in GraphGeneration.jl/src/GraphGeneration.jl
- [X] T032 [US1] Implement make_hypergraph_from_text function with caching in GraphGeneration.jl/src/GraphGeneration.jl
- [X] T033 [US1] Implement CLI interface using ArgParse.jl in GraphGeneration.jl/src/GraphGeneration.jl (cli_main function)
- [X] T034 [US1] Add error handling and validation to all GraphGeneration.jl functions
- [X] T035 [US1] Add structured logging to GraphGeneration.jl operations
- [X] T036 [US1] Create Documenter.jl documentation structure in GraphGeneration.jl/docs/src/
- [X] T037 [US1] Write API documentation for GraphGeneration.jl functions in GraphGeneration.jl/docs/src/index.md
- [X] T038 [US1] Build Documenter.jl documentation for GraphGeneration.jl (verify HTML output)

### Notebook Conversion for User Story 1

- [X] T039 [US1] Create notebooks/ directory structure at repository root (mirroring HyperGraphReasoningPython/Notebooks/SG/ structure)
- [X] T040 [P] [US1] Write test to verify notebook conversion: test that notebooks/hypergraph-reasoning/sg/make_hypergraph.jl exists and is valid Pluto notebook format
- [X] T041 [P] [US1] Write test to verify notebook conversion: test that notebooks/hypergraph-reasoning/sg/Analyze_Hypergraph.jl exists and is valid Pluto notebook format
- [X] T042 [P] [US1] Write test to verify notebook conversion: test that notebooks/hypergraph-reasoning/sg/Agents.jl exists and is valid Pluto notebook format
- [X] T043 [P] [US1] Write test for notebook execution: test that converted notebooks can be executed in Pluto.jl without errors
- [X] T044 [P] [US1] Write test for notebook output equivalence: automated comparison of notebook outputs with Python reference data (where applicable)
- [X] T045 [US1] Convert make_hypergraph.ipynb to Pluto notebook: translate Python code to Julia, replace imports with `using` statements for the top-level Julia packages, store in notebooks/hypergraph-reasoning/sg/make_hypergraph.jl
- [X] T046 [US1] Convert Analyze_Hypergraph.ipynb to Pluto notebook: translate Python code to Julia, replace imports with `using` statements for the top-level Julia packages, store in notebooks/hypergraph-reasoning/sg/Analyze_Hypergraph.jl
- [X] T047 [US1] Convert Agents.ipynb to Pluto notebook: translate Python code to Julia, replace imports with `using` statements for the top-level Julia packages, store in notebooks/hypergraph-reasoning/sg/Agents.jl
- [X] T048 [US1] Verify all converted notebooks execute successfully in Pluto.jl: manual execution and verification of each notebook
- [X] T049 [US1] Validate notebook output equivalence: compare outputs from converted notebooks with Python notebook outputs (automated where possible, manual for visual/interactive outputs)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. GraphGeneration.jl package can generate hypergraphs from text with functional equivalence to Python implementation. All 3 Python notebooks are converted to Pluto notebooks and execute successfully with equivalent outputs.

---

## Phase 4: User Story 1 (Continued) - Notebook Conversion

**Note**: Notebook conversion tasks are part of User Story 1 as they depend on the top-level Julia packages being implemented first.

---

## Phase 5: User Story 2 - Maintain Typst Documentation (Priority: P2)

**Goal**: Create Typst documentation in `docs/hypergraph-reasoning/` covering at least one replicated module (GraphGeneration.jl) with progress notes, implementation approach, and acquired knowledge

**Independent Test**: Create Typst document for GraphGeneration.jl replication, compile to PDF successfully, verify content includes progress, implementation details, and learnings

### Tests for User Story 2 (REQUIRED - TDD mandatory)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T050 [P] [US2] Write test to verify Typst file exists in docs/hypergraph-reasoning/graph-generation.typ
- [ ] T051 [P] [US2] Write test to verify Typst compilation succeeds (test script in docs/hypergraph-reasoning/test/verify_typst.jl or shell script)

### Implementation for User Story 2

- [ ] T052 [US2] Create Typst document structure in docs/hypergraph-reasoning/graph-generation.typ
- [ ] T053 [US2] Write progress section documenting GraphGeneration.jl replication status in docs/hypergraph-reasoning/graph-generation.typ
- [ ] T054 [US2] Write implementation section describing design decisions and Julia-idiomatic approaches in docs/hypergraph-reasoning/graph-generation.typ
- [ ] T055 [US2] Write learnings section capturing challenges encountered and solutions adopted in docs/hypergraph-reasoning/graph-generation.typ
- [ ] T056 [US2] Add code examples and comparisons (Python vs Julia) in docs/hypergraph-reasoning/graph-generation.typ
- [ ] T057 [US2] Compile Typst document to PDF: docs/hypergraph-reasoning/graph-generation.pdf
- [ ] T058 [US2] Verify Typst compilation produces well-formatted output

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Typst documentation exists and compiles successfully.

---

## Phase 6: User Story 3 - Maintain Development Standards via AGENTS.md (Priority: P3)

**Goal**: Ensure AGENTS.md exists with at least 5 distinct development standards or conventions that guide Julia package development

**Independent Test**: Verify AGENTS.md exists, contains at least 5 distinct standards, and is readable/accessible to developers

### Tests for User Story 3 (REQUIRED - TDD mandatory)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T059 [P] [US3] Write test to verify AGENTS.md exists at repository root
- [ ] T060 [P] [US3] Write test to verify AGENTS.md contains at least 5 distinct development standards

### Implementation for User Story 3

- [ ] T061 [US3] Create or update AGENTS.md at repository root with Julia package development standards
- [ ] T062 [US3] Document Julia package structure conventions in AGENTS.md
- [ ] T063 [US3] Document Julia coding standards and best practices in AGENTS.md
- [ ] T064 [US3] Document TDD workflow and testing conventions in AGENTS.md
- [ ] T065 [US3] Document CLI interface requirements and patterns in AGENTS.md
- [ ] T066 [US3] Document documentation standards (Documenter.jl + Typst) in AGENTS.md
- [ ] T067 [US3] Document dependency management and Project.toml conventions in AGENTS.md
- [ ] T068 [US3] Document notebook conversion standards (Pluto notebooks) in AGENTS.md
- [ ] T069 [US3] Verify AGENTS.md contains at least 5 distinct standards

**Checkpoint**: All three user stories should now be independently functional. AGENTS.md provides clear development guidance.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T070 [P] Update all package README.md files with installation and usage instructions
- [ ] T071 [P] Add semantic versioning tags to all packages (start at v0.1.0)
- [ ] T072 [P] Run quickstart.md validation: verify all examples work correctly
- [ ] T073 Code cleanup and refactoring across all packages
- [ ] T074 [P] Performance optimization: profile and optimize critical paths in GraphGeneration.jl
- [ ] T075 [P] Add additional unit tests for edge cases in GraphGeneration.jl/test/
- [ ] T076 Verify all packages can be installed independently via Pkg.add()
- [ ] T077 Verify inter-package dependencies resolve correctly (GraphGeneration.jl depending on GraphUtils.jl)
- [ ] T078 Verify all converted Pluto notebooks can be opened and executed in Pluto.jl
- [ ] T079 Update Typst documentation with additional modules as they are replicated
- [ ] T080 Update Typst documentation with notebook conversion progress and learnings
- [ ] T081 Update AGENTS.md with any new patterns discovered during implementation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories. Requires GraphUtils.jl (T022-T024) before GraphGeneration.jl implementation. Notebook conversion (T039-T049) depends on GraphGeneration.jl being implemented first (notebooks use the top-level Julia packages).
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on User Story 1 completion (needs replicated module to document)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Can proceed independently, but benefits from patterns discovered in US1

### Within Each User Story

- Tests (REQUIRED) MUST be written and FAIL before implementation
- GraphUtils.jl must be implemented before GraphGeneration.jl (dependency: T022-T026 before T028)
- Core functions before CLI interface
- Core implementation before documentation
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T002-T006)
- All Foundational dependency tasks marked [P] can run in parallel (T008-T014)
- GraphUtils.jl implementation (T022-T026) can proceed in parallel with test writing (T017-T021)
- All test tasks for a user story marked [P] can run in parallel (T017-T021, T040-T044)
- Notebook conversion tasks (T045-T047) can run in parallel (different notebooks)
- User Stories 2 and 3 can start in parallel after US1 completes (though US2 benefits from US1 completion)
- Polish phase tasks marked [P] can run in parallel (T070, T071, T074, T075)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Write equivalence test for documents2dataframe in GraphGeneration.jl/test/test_documents2dataframe.jl"
Task: "Write equivalence test for df2hypergraph in GraphGeneration.jl/test/test_df2hypergraph.jl"
Task: "Write equivalence test for make_hypergraph_from_text in GraphGeneration.jl/test/test_make_hypergraph.jl"
Task: "Write integration test for end-to-end hypergraph generation workflow in GraphGeneration.jl/test/test_integration.jl"
Task: "Write CLI interface test in GraphGeneration.jl/test/test_cli.jl"

# Launch GraphUtils.jl implementation in parallel with tests:
Task: "Create GraphUtils.jl module structure in GraphUtils.jl/src/GraphUtils.jl"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (GraphGeneration.jl package + notebook conversion)
4. **STOP and VALIDATE**: Test User Story 1 independently - verify hypergraph generation works with functional equivalence AND converted notebooks execute successfully
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 (packages + notebooks) → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo (Documentation)
4. Add User Story 3 → Test independently → Deploy/Demo (Standards)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (GraphGeneration.jl)
   - Developer B: User Story 2 (Typst documentation) - can start after US1 completes
   - Developer C: User Story 3 (AGENTS.md) - can start independently
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD mandatory per Constitution)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- GraphUtils.jl is a dependency for GraphGeneration.jl - implement first (T022-T026)
- All packages must include CLI interfaces per Constitution Principle II
  - GraphUtils.jl CLI: T025-T026 (included in US1)
  - GraphGeneration.jl CLI: T021, T033 (included in US1)
  - GraphAnalysis.jl CLI: Will be added when GraphAnalysis.jl implementation phase is created
  - GraphTools.jl CLI: Will be added when GraphTools.jl implementation phase is created
- Tests must use reference data fixtures from Python for equivalence validation
- Notebook conversion (T039-T049) requires GraphGeneration.jl to be implemented first (notebooks use the top-level Julia packages)
- Pluto notebooks specify dependencies via `using` statements within notebook files (Pluto manages environment automatically)
