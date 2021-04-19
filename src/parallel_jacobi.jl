"""
$(TYPEDEF)

Parallel Jacobi preconditioner
"""
mutable struct ParallelJacobiPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    invdiag::Array{Tv,1}
    function ParallelJacobiPreconditioner{Tv,Ti}() where {Tv,Ti}
        p=new()
        p.invdiag=zeros(Tv,0)
        p
    end
end

function update!(precon::ParallelJacobiPreconditioner{Tv,Ti}) where {Tv,Ti}
    cscmatrix=precon.A.cscmatrix
    if length(precon.invdiag)==0
        precon.invdiag=Array{Tv,1}(undef,cscmatrix.n)
    end
    invdiag=precon.invdiag
    n=cscmatrix.n
    Threads.@threads for i=1:n
        @inbounds  invdiag[i]=1.0/cscmatrix[i,i]
    end
    precon
end

ParallelJacobiPreconditioner()=ParallelJacobiPreconditioner{Float64,Int64}()


"""
```
ParallelJacobiPreconditioner(A)
ParallelJacobiPreconditioner(cscmatrix)
```
"""
ParallelJacobiPreconditioner(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}=factorize!(ParallelJacobiPreconditioner{Tv, Ti}(),A)
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
