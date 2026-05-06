# GraphUtils.jl

Utility functions for text processing and extraction.

## Installation

```julia
using Pkg
Pkg.add("GraphUtils")
```

Or from local path:

```julia
Pkg.add(path="replica/GraphUtils")
```

## Usage

```julia
using GraphUtils

# Extract substring between delimiters
extract("[hello] world [test]")  # Returns "[hello] world [test]"

# Remove markdown symbols
remove_markdown_symbols("**bold** text")  # Returns "bold text"
```

## CLI Usage

```bash
# Extract command
echo "[test]" | julia --project=. -e 'using GraphUtils; GraphUtils.cli_main(["extract"])'

# Remove markdown
echo "**bold**" | julia --project=. -e 'using GraphUtils; GraphUtils.cli_main(["remove-markdown"])'
```

## Development

Run tests:

```julia
using Pkg
Pkg.test("GraphUtils")
```

## License

See repository LICENSE file.
