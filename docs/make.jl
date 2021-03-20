push!(LOAD_PATH,"../src/")
using Documenter, ExtendableSparse

makedocs(sitename="ExtendableSparse.jl",
         modules = [ExtendableSparse],
         doctest = true,
         clean = false,
         authors = "J. Fuhrmann",
         repo="https://github.com/j-fu/ExtendableSparse.jl",
         pages=[
             "Home"=>"index.md",
             "example.md",
             "extsparse.md",
             "iter.md",
             "changes.md",
         ])

deploydocs(repo = "github.com/j-fu/ExtendableSparse.jl.git")
