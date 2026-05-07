using Documenter
using GraphGeneration

makedocs(
    sitename="GraphGeneration.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    modules=[GraphGeneration],
)

deploydocs(
    repo="github.com/yourusername/GraphGeneration.jl.git",
)
