push!(LOAD_PATH,"../src/")
using Documenter, ExtendableSparse
println(pwd())
makedocs(
    sitename="ExtendableSparse.jl",
    modules = [ExtendableSparse],
    clean = true,
    authors = "J. Fuhrmann",
    version = "0.1.0",
    pages=[
        "Home"=>"index.md",
        "changes.md",
        "api.md"
    ]
)

