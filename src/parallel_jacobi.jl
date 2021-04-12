"""
$(TYPEDEF)

Parallel Jacobi preconditioner
"""
mutable struct ParallelJacobiPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    invdiag::Array{Tv,1}
end

function update!(precon::ParallelJacobiPreconditioner)
    cscmatrix=precon.A.cscmatrix
    invdiag=precon.invdiag
    n=cscmatrix.n
    Threads.@threads for i=1:n
        @inbounds  invdiag[i]=1.0/cscmatrix[i,i]
    end
    precon
end

"""
```
ParallelJacobiPreconditioner(A)
ParallelJacobiPreconditioner(cscmatrix)
```
"""
function ParallelJacobiPreconditioner(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @assert size(A,1)==size(A,2)
    flush!(A)
    invdiag=Array{Tv,1}(undef,A.cscmatrix.n)
    precon=JacobiPreconditioner{Tv, Ti}(A,invdiag)
    update!(precon)
end

ParallelJacobiPreconditioner(cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}=ParallelJacobiPreconditioner(ExtendableSparseMatrix(cscmatrix))

function  LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, precon::ParallelJacobiPreconditioner, v::AbstractArray{T,1} where T)
    invdiag=precon.invdiag
    n=length(invdiag)
    Threads.@threads for i=1:n
        @inbounds u[i]=invdiag[i]*v[i]
    end
end

function LinearAlgebra.ldiv!(precon::ParallelJacobiPreconditioner, v::AbstractArray{T,1} where T)
    ldiv!(v, precon, v)
end
