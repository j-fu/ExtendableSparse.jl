
mutable struct SparspakLU{Tv, Ti} <: AbstractLUFactorization{Tv, Ti}
    A::Union{ExtendableSparseMatrix{Tv, Ti}, Nothing}
    S::Union{Sparspak.SparseSolver.SparseSolver{Ti, Tv}, Nothing}
    phash::UInt64
end

function SparspakLU{Tv, Ti}() where {Tv, Ti}
    fact = SparspakLU{Tv, Ti}(nothing, nothing, 0)
end

"""
```
SparspakLU(;valuetype=Float64) 
```

LU factorization based on [Sparspak.jl](https://github.com/PetrKryslUCSD/Sparspak.jl) (P.Krysl's Julia re-implementation of Sparspak by George & Liu)
"""
function SparspakLU(; valuetype::Type = Float64, indextype::Type = Int64, kwargs...)
    SparspakLU{valuetype, indextype}(; kwargs...)
end

function update!(lufact::SparspakLU{Tv, Ti}) where {Tv, Ti}
    flush!(lufact.A)
    if lufact.A.phash != lufact.phash
        lufact.S = sparspaklu(lufact.A.cscmatrix)
    else
        sparspaklu!(lufact.S, lufact.A.cscmatrix)
    end
end

LinearAlgebra.ldiv!(u, lufact::SparspakLU, v) = ldiv!(u, lufact.S, v)
LinearAlgebra.ldiv!(lufact::SparspakLU, v) = ldiv!(lufact.S, v)
