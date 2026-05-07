# Development Standards and Guidelines

This document maintains development standards, coding conventions, and project best practices for the Julia replication project.

## Julia Package Structure Conventions

### Package Organization

- **One package per Python module**: Each Python module (`graph_generation.py`, `graph_analysis.py`, etc.) maps to one Julia package (`GraphGeneration.jl`, `GraphAnalysis.jl`, etc.)
- **Package location**: All Julia packages live as top-level `*.jl/` directories at repository root
- **Standard structure**: Each package follows Julia standard structure:
  ```
  PackageName.jl/
  ├── src/
  │   └── PackageName.jl
  ├── test/
  │   ├── test_*.jl
  │   └── test_helpers.jl
  ├── docs/
  │   ├── src/
  │   └── make.jl
  └── Project.toml
  ```

### Package Naming

- Use PascalCase: `GraphGeneration.jl`, `GraphAnalysis.jl`
- Match Python module names where possible (camelCase → PascalCase)
- Keep names descriptive and consistent

### Project.toml Requirements

- **Version**: Start at `0.1.0` (semantic versioning)
- **Dependencies**: Explicitly list all dependencies with UUIDs
- **Extras**: Use `[extras]` section for optional dependencies (e.g., test-only deps)
- **Documenter**: Include `Documenter` in dependencies if package has docs

## Julia Coding Standards and Best Practices

### Type Annotations

- **Function signatures**: Always annotate return types for performance and clarity
  ```julia
  function my_function(x::String)::DataFrame
      # ...
  end
  ```
- **Type stability**: Ensure functions are type-stable for optimal performance
- **Abstract types**: Use abstract types for flexibility where appropriate

### Error Handling

- **Exception types**: Use appropriate Julia exception types:
  - `ArgumentError`: Invalid function arguments
  - `KeyError`: Missing dictionary/collection keys
  - `IOError`: File/system I/O errors
  - `ErrorException`: Custom errors with descriptive messages
- **Error messages**: Provide clear, actionable error messages
- **Logging**: Use structured logging for errors (via `LoggingUtils`)

### Code Style

- **Indentation**: 4 spaces (Julia standard)
- **Line length**: Aim for <100 characters, wrap when necessary
- **Naming**:
  - Functions: `snake_case` (Julia convention)
  - Types: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Documentation**: Use docstrings for all exported functions
  ```julia
  """
      my_function(x::String) -> DataFrame

  Description of what the function does.

  # Arguments
  - `x::String`: Description of argument

  # Returns
  - `DataFrame`: Description of return value
  """
  ```

### Performance

- **Avoid global variables**: Use function parameters and return values
- **Pre-allocate**: Pre-allocate arrays when size is known
- **Type stability**: Ensure functions are type-stable
- **Profiling**: Profile critical paths before optimizing

## TDD Workflow and Testing Conventions

### Test-Driven Development (MANDATORY)

Per Constitution Principle III, TDD is **NON-NEGOTIABLE**:

1. **Write tests FIRST**: Tests must be written before implementation
2. **Tests must FAIL**: Verify tests fail before implementing
3. **Implement**: Write minimal code to make tests pass
4. **Refactor**: Improve code while keeping tests green

### Test Organization

- **Test files**: One test file per function/module: `test_functionname.jl`
- **Test helpers**: Shared utilities in `test/test_helpers.jl`
- **Test fixtures**: Reference data in `test/fixtures/`
- **Test structure**: Use `@testset` blocks for organization
  ```julia
  @testset "function_name tests" begin
      @testset "basic functionality" begin
          # tests
      end
      @testset "edge cases" begin
          # tests
      end
  end
  ```

### Equivalence Testing

- **Python reference data**: Generate reference outputs from Python implementation
- **Comparison utilities**: Use `test_helpers.jl` for comparing hypergraphs, embeddings, DataFrames
- **Tolerance**: Use appropriate numerical tolerance for floating-point comparisons (default: 1e-6)
- **Manual verification**: For complex/edge cases, supplement automated tests with manual verification

### Test Coverage

- **Unit tests**: Test each function independently
- **Integration tests**: Test end-to-end workflows
- **CLI tests**: Test command-line interfaces
- **Error cases**: Test error handling and edge cases

## CLI Interface Requirements and Patterns

### Constitution Requirement

Per Constitution Principle II, **every library MUST expose CLI functionality**.

### Implementation Pattern

- **Use ArgParse.jl**: Standard library for argument parsing
- **Function signature**: `cli_main(args::Vector{String}=ARGS)`
- **Text I/O**: stdin → stdout, errors → stderr
- **Output formats**: Support both JSON and human-readable formats
- **Help text**: Always provide `--help` option

### Example Structure

```julia
function cli_main(args::Vector{String}=ARGS)
    parser = ArgParseSettings(
        description="Package description",
        version="0.1.0",
        add_version=true
    )
    
    @add_arg_table! parser begin
        "--input"
            help = "Input file or stdin"
            default = ""
        "--output"
            help = "Output file or stdout"
            default = ""
    end
    
    parsed_args = parse_args(args, parser)
    # ... implementation
    return 0
end
```

## Documentation Standards

### Dual Documentation System

1. **Documenter.jl**: API documentation within each package
   - Location: `docs/src/`
   - Build: `julia docs/make.jl`
   - Output: HTML documentation
   - Required for all packages

2. **Typst**: Global progress and learnings documentation
   - Location: `documentation/`
   - Format: `.typ` files
   - Content: Progress, implementation details, design decisions, learnings
   - Compilation: `typst compile file.typ file.pdf`

### Documenter.jl Structure

- **Main page**: `docs/src/index.md`
- **Module docs**: Use `@docs` blocks for automatic API documentation
- **Examples**: Include usage examples in documentation
- **Build script**: `docs/make.jl` using Documenter.jl

### Typst Documentation

- **Progress tracking**: Document implementation status
- **Design decisions**: Explain Julia-idiomatic choices
- **Learnings**: Capture challenges and solutions
- **Code comparisons**: Python vs Julia code examples
- **Next steps**: Future work and improvements

## Dependency Management and Project.toml Conventions

### Dependency Declaration

- **Explicit UUIDs**: Always include UUIDs for dependencies
- **Version constraints**: Use compatible version ranges when needed
- **Standard libraries**: Use standard library modules (e.g., `Test`, `SHA`) without UUID

### Local Dependencies

- **Development**: During development, use local includes or path-based dependencies
- **Production**: For production, register packages and use proper Pkg dependencies
- **GraphUtils dependency**: GraphGeneration.jl depends on GraphUtils.jl (add to `[extras]` or proper dependency)

### Dependency Categories

- **Core dependencies**: Required for package functionality
- **Test dependencies**: Only needed for testing (use `[extras]` or test-only Project.toml)
- **Documentation dependencies**: Only needed for building docs (e.g., `Documenter`)

### Example Project.toml

```toml
[deps]
Graphs = "86223c79-3864-5bf0-83f7-82e325a511b8"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[extras]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
```

## Version Control and Git Workflow

### Commit Messages

- **Format**: `type: description`
- **Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- **Examples**:
  - `feat: implement documents2dataframe function`
  - `test: add equivalence tests for df2hypergraph`
  - `docs: update API documentation`

### Branch Strategy

- **Feature branches**: `###-feature-name` (e.g., `001-julia-replication`)
- **Main branch**: `main` (protected, requires PR)

## Code Review Checklist

Before submitting code for review, ensure:

- [ ] All tests pass
- [ ] TDD workflow followed (tests written first)
- [ ] CLI interface implemented (if applicable)
- [ ] Documentation updated (Documenter.jl and/or Typst)
- [ ] Error handling implemented
- [ ] Logging added for key operations
- [ ] Type annotations present
- [ ] Code follows style guidelines
- [ ] No hardcoded paths or secrets
- [ ] Dependencies properly declared in Project.toml

## Additional Resources

- **Julia Style Guide**: https://docs.julialang.org/en/v1/manual/style-guide/
- **Julia Package Development**: https://pkgdocs.julialang.org/
- **Documenter.jl**: https://documenter.juliadocs.org/
- **ArgParse.jl**: https://argparsejl.readthedocs.io/

---

**Last Updated**: 2026-01-16  
**Version**: 1.0.0
