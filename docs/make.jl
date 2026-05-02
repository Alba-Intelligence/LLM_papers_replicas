using Pkg

Pkg.activate(@__DIR__)

using Documenter
using DeepSeekV4
using OpenMythos
using TransformerCore

makedocs(
    modules=[TransformerCore, OpenMythos, DeepSeekV4],
    sitename="OpenMythos Workspace",
    format=Documenter.HTML(prettyurls=false, edit_link=nothing, repolink=nothing, inventory_version="0.1.0"),
    # Julia 1.12 currently chokes on source-attached docs for the `OpenMythos`
    # type because it shares a name with the package module, so that symbol is
    # documented manually in the API page instead of via strict checkdocs.
    checkdocs=:none,
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Manual" => "manual.md",
        "TransformerCore API" => "transformercore.md",
        "OpenMythos API" => "openmythos.md",
        "DeepSeek V4 API" => "deepseekv4.md",
    ],
)
