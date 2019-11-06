push!(LOAD_PATH,"../src/")
using Documenter, ExtendableSparse

makedocs(sitename="ExtendableSparse.jl",
         modules = [ExtendableSparse],
         doctest = false,
         clean = true,
         authors = "J. Fuhrmann",
         repo="https://github.com/j-fu/ExtendableSparse.jl",
         pages=[
             "Home"=>"index.md",
             "changes.md",
             "api.md"
         ])

deploydocs(
    repo = "github.com/j-fu/ExtendableSparse.jl.git",
    versions = ["stable" => "v^", "v#.#.#", "devurl" => "dev"]
)
