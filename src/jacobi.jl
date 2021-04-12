"""
$(TYPEDEF)

Jacobi preconditoner
"""
mutable struct JacobiPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    invdiag::Array{Tv,1}
end

function update!(precon::JacobiPreconditioner)
    cscmatrix=precon.A.cscmatrix
    invdiag=precon.invdiag
    n=cscmatrix.n
    @inbounds for i=1:n
        invdiag[i]=1.0/cscmatrix[i,i]
    end
    precon
end

"""
```
JacobiPreconditioner(A)
JacobiPreconditioner(cscmatrix)
```
"""
function JacobiPreconditioner(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @assert size(A,1)==size(A,2)
    flush!(A)
    invdiag=Array{Tv,1}(undef,A.cscmatrix.n)
    precon=JacobiPreconditioner{Tv, Ti}(A,invdiag)
    update!(precon)
end

JacobiPreconditioner(cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}=JacobiPreconditioner(ExtendableSparseMatrix(cscmatrix))


function  LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, precon::JacobiPreconditioner, v::AbstractArray{T,1} where T)
    invdiag=precon.invdiag
    n=length(invdiag)
    @inbounds @simd for i=1:n
        u[i]=invdiag[i]*v[i]
    end
end

function LinearAlgebra.ldiv!(precon::JacobiPreconditioner, v::AbstractArray{T,1} where T)
    ldiv!(v, precon, v)
end
