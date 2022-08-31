mutable struct SparspakLU{Tv, Ti} <: AbstractLUFactorization{Tv,Ti} 
    A::Union{ExtendableSparseMatrix{Tv,Ti},Nothing}
    P::Union{Sparspak.SpkProblem.Problem{Ti,Tv},Nothing}
    S::Union{Sparspak.SparseSolver.SparseSolver{Ti,Tv},Nothing}
    phash::UInt64
end

function SparspakLU{Tv,Ti}() where {Tv,Ti}
    fact=SparspakLU{Tv,Ti}(nothing,nothing,nothing,0)
end


"""
```
SparspakLU(;valuetype=Float64) 
```

LU factorization based on P.Kryls's re-implementation of SPARSPAK.
"""
SparspakLU(;valuetype::Type=Float64, indextype::Type=Int64,kwargs...)=SparspakLU{valuetype,indextype}(;kwargs...)

function update!(lufact::SparspakLU{Tv,Ti}) where {Tv, Ti}
    flush!(lufact.A)
    lufact.P=Sparspak.SpkProblem.Problem(size(lufact.A)...)
    Sparspak.SpkProblem.insparse!(lufact.P,lufact.A)
    lufact.S=Sparspak.SparseSolver.SparseSolver(lufact.P)
end

function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, lufact::SparspakLU, v::AbstractArray{T,1} where T)
    u.=ldiv!(lufact,v)
end

function LinearAlgebra.ldiv!(lufact::SparspakLU, v::AbstractArray{T,1} where T)
    Sparspak.SpkProblem.infullrhs!(lufact.P,v)
    Sparspak.SparseSolver.solve!(lufact.S)
    lufact.P.x
end


@eval begin
    @makefrommatrix SparspakLU
end
