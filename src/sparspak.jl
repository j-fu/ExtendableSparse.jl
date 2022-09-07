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

LU factorization based on [Sparspak.jl](https://github.com/PetrKryslUCSD/Sparspak.jl) (P.Krysl's Julia re-implementation of Sparspak by George & Liu)
"""
SparspakLU(;valuetype::Type=Float64, indextype::Type=Int64,kwargs...)=SparspakLU{valuetype,indextype}(;kwargs...)

function update!(lufact::SparspakLU{Tv,Ti}) where {Tv, Ti}
    flush!(lufact.A)
    lufact.P=Sparspak.SpkProblem.Problem(size(lufact.A)...,nnz(lufact.A),zero(Tv))
    Sparspak.SpkProblem.insparse!(lufact.P,lufact.A)
    lufact.S=Sparspak.SparseSolver.SparseSolver(lufact.P)
end

function LinearAlgebra.ldiv!(u::AbstractArray{Tu,1}, lufact::SparspakLU{T, Ti}, v::AbstractArray{Tv,1}) where {T,Tu,Tv,Ti}
    u.=ldiv!(lufact,v)
end

function LinearAlgebra.ldiv!(lufact::SparspakLU{T, Ti}, v::AbstractArray{Tv,1} ) where {T,Tv,Ti}
    Sparspak.SpkProblem.infullrhs!(lufact.P,T.(v))
    Sparspak.SparseSolver.solve!(lufact.S)
    lufact.P.x
end


@eval begin
    @makefrommatrix SparspakLU
end
