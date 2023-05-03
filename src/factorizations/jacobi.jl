mutable struct _JacobiPreconditioner{Tv}
    invdiag::Vector{Tv}
end

function jacobi(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    invdiag = Array{Tv, 1}(undef, A.n)
    n = A.n
    @inbounds for i = 1:n
        invdiag[i] = one(Tv) / A[i, i]
    end
    _JacobiPreconditioner(invdiag)
end

function jacobi!(p::_JacobiPreconditioner{Tv},A::SparseMatrixCSC{Tv,Ti})where {Tv,Ti}
    n = A.n
    @inbounds for i = 1:n
        p.invdiag[i] = one(Tv) / A[i, i]
    end
    p
end

mutable struct JacobiPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::Union{_JacobiPreconditioner,Nothing}
    phash::UInt64
    function JacobiPreconditioner()
        p = new()
        p.factorization=nothing
        p.phash=0
        p
    end
end

function LinearAlgebra.ldiv!(u,p::_JacobiPreconditioner,v)
    n=length(p.invdiag)
    for i = 1:n
        @inbounds u[i] = p.invdiag[i] * v[i]
    end
end

 LinearAlgebra.ldiv!(p::_JacobiPreconditioner,v)=ldiv!(v,p,v)


"""
```
JacobiPreconditioner()
JacobiPreconditioner(matrix)
```

Jacobi preconditioner.
"""
function JacobiPreconditioner end

function update!(p::JacobiPreconditioner)
    flush!(p.A)
    Tv=eltype(p.A)
    if p.A.phash!=p.phash || isnothing(p.factorization)
        p.factorization=jacobi(p.A.cscmatrix)
        p.phash=p.A.phash
    else
        jacobi!(p.factorization,p.A.cscmatrix)
    end
    p
end

allow_views(::JacobiPreconditioner)=true
allow_views(::Type{JacobiPreconditioner})=true
