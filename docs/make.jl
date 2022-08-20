push!(LOAD_PATH,"../src/")
using Documenter, ExtendableSparse,Pardiso,AlgebraicMultigrid,IncompleteLU,LinearSolve

function mkdocs()
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
                 "linearsolve.md",
                 "internal.md",
                 "changes.md",
             ])
end

mkdocs()

deploydocs(repo = "github.com/j-fu/ExtendableSparse.jl.git")
