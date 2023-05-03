mutable struct _ParallelJacobiPreconditioner{Tv}
    invdiag::Vector{Tv}
end

function pjacobi(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    invdiag = Array{Tv, 1}(undef, A.n)
    n = A.n
    Threads.@threads for i = 1:n
        invdiag[i] = one(Tv) / A[i, i]
    end
    _ParallelJacobiPreconditioner(invdiag)
end

function pjacobi!(p::_ParallelJacobiPreconditioner{Tv},A::SparseMatrixCSC{Tv,Ti})where {Tv,Ti}
    n = A.n
    Threads.@threads for i = 1:n
        p.invdiag[i] = one(Tv) / A[i, i]
    end
    p
end


function LinearAlgebra.ldiv!(u,p::_ParallelJacobiPreconditioner,v)
    n=length(p.invdiag)
    for i = 1:n
        @inbounds u[i] = p.invdiag[i] * v[i]
    end
end

LinearAlgebra.ldiv!(p::_ParallelJacobiPreconditioner,v)=ldiv!(v,p,v)


mutable struct ParallelJacobiPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::Union{_ParallelJacobiPreconditioner,Nothing}
    phash::UInt64
    function ParallelJacobiPreconditioner()
        p = new()
        p.factorization = nothing
        p.phash=0
        p
    end
end

"""
```
ParallelJacobiPreconditioner()
ParallelJacobiPreconditioner(matrix)
```

ParallelJacobi preconditioner.
"""
function ParallelJacobiPreconditioner end

function update!(p::ParallelJacobiPreconditioner)
    flush!(p.A)
    Tv=eltype(p.A)
    if p.A.phash!=p.phash || isnothing(p.factorization)
        p.factorization=pjacobi(p.A.cscmatrix)
        p.phash=p.A.phash
    else
        pjacobi!(p.factorization,p.A.cscmatrix)
    end
    p
end

allow_views(::ParallelJacobiPreconditioner)=true
allow_views(::Type{ParallelJacobiPreconditioner})=true
