using Documenter
using RobustMotionDeblur

makedocs(
    sitename = "RobustMotionDeblur.jl",
    modules = [RobustMotionDeblur],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "Algorithm" => "algorithm.md",
        "API" => "api.md",
        "Developers" => "developers.md",
    ],
)
