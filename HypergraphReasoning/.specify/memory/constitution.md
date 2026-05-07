<!--
Sync Impact Report:
Version: (none) → 1.0.0
Modified Principles: N/A (initial version)
Added Sections: Core Principles (5), Development Workflow, Governance
Removed Sections: N/A
Templates Requiring Updates:
  ✅ .specify/templates/plan-template.md - Constitution Check section aligns with principles
  ✅ .specify/templates/spec-template.md - No changes needed (user stories already mandatory)
  ✅ .specify/templates/tasks-template.md - Task structure aligns with test-first principle
Follow-up TODOs: None
-->

# Hypergraph Constitution

## Core Principles

### I. Library-First

Every feature starts as a standalone Julia package. Libraries MUST be self-contained,
independently testable, and documented. Each library requires a clear purpose—no
organizational-only libraries. Packages MUST be usable independently of the main project
structure.

**Rationale**: Enables modular development, independent testing, and reuse across
different contexts. Supports the Julia ecosystem's package-oriented architecture.

### II. CLI Interface

Every library exposes functionality via command-line interface. Text in/out protocol:
stdin/args → stdout, errors → stderr. Support both JSON and human-readable formats.
CLI tools MUST be executable as standalone binaries or via `julia --project`.

**Rationale**: Enables integration with other tools, automation, and debugging. Text
I/O ensures debuggability and composability with Unix tools.

### III. Test-First (NON-NEGOTIABLE)

Test-Driven Development is mandatory. Process: Tests written → User approved → Tests
fail → Then implement. Red-Green-Refactor cycle strictly enforced. No implementation
code without corresponding tests.

**Rationale**: Ensures correctness, prevents regressions, and documents expected
behavior. Tests serve as executable specifications.

### IV. Integration Testing

Focus areas requiring integration tests: New library contract tests, Contract changes,
Inter-service communication, Shared schemas. Integration tests MUST verify end-to-end
workflows and cross-component interactions.

**Rationale**: Unit tests verify components in isolation; integration tests verify
system behavior when components interact. Critical for scientific computing where
data flows through multiple transformations.

### V. Observability, Versioning & Simplicity

Text I/O ensures debuggability. Structured logging required for all operations.
Versioning follows Semantic Versioning (MAJOR.MINOR.PATCH). Start simple—YAGNI
principles apply. Complexity MUST be justified in documentation.

**Rationale**: Debuggability is essential for research reproducibility. Versioning
enables dependency management. Simplicity reduces maintenance burden and cognitive
load.

## Development Workflow

All PRs and reviews MUST verify constitution compliance. Code reviews check:
constitution adherence, test coverage, documentation completeness, and complexity
justification. Use `.specify/templates/agent-file-template.md` for runtime development
guidance.

## Governance

This constitution supersedes all other development practices. Amendments require:
documentation of rationale, approval process, and migration plan for existing code.
All changes MUST increment the version number according to semantic versioning rules.

**Amendment Procedure**: Propose changes via PR with clear rationale. Changes affecting
core principles require broader review. Version bump type (MAJOR/MINOR/PATCH) determined
by impact scope.

**Compliance Review**: Every feature specification and implementation plan MUST include
a Constitution Check section verifying adherence to all applicable principles.

**Version**: 1.0.0 | **Ratified**: 2026-01-16 | **Last Amended**: 2026-01-16
