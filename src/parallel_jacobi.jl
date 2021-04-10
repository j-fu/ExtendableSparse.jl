"""
$(TYPEDEF)

Parallel Jacobi preconditioner
"""
struct ParallelJacobiPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    extmatrix::ExtendableSparseMatrix{Tv,Ti}
    invdiag::Array{Tv,1}
end

"""
$(SIGNATURES)

Parallel Jacobi preconditioner update
"""
function update!(precon::ParallelJacobiPreconditioner)
    cscmatrix=precon.extmatrix.cscmatrix
    invdiag=precon.invdiag
    n=cscmatrix.n
    Threads.@threads for i=1:n
        @inbounds  invdiag[i]=1.0/cscmatrix[i,i]
    end
    precon
end

"""
$(SIGNATURES)

Parallel Jacobi preconditioner construtor
"""
function ParallelJacobiPreconditioner(extmatrix::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @assert size(extmatrix,1)==size(extmatrix,2)
    flush!(extmatrix)
    invdiag=Array{Tv,1}(undef,extmatrix.cscmatrix.n)
    precon=JacobiPreconditioner{Tv, Ti}(extmatrix,invdiag)
    update!(precon)
end

"""
$(SIGNATURES)

Parallel Jacobi preconditioner construtor
"""
ParallelJacobiPreconditioner(cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}=ParallelJacobiPreconditioner(ExtendableSparseMatrix(cscmatrix))


"""
$(SIGNATURES)

Parallel Jacobi preconditioner solve
"""
function  LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, precon::ParallelJacobiPreconditioner, v::AbstractArray{T,1} where T)
    invdiag=precon.invdiag
    n=length(invdiag)
    Threads.@threads for i=1:n
        @inbounds u[i]=invdiag[i]*v[i]
    end
end

"""
$(SIGNATURES)

Parallel Jacobi preconditioner inplace solve
"""
function LinearAlgebra.ldiv!(precon::ParallelJacobiPreconditioner, v::AbstractArray{T,1} where T)
    ldiv!(v, precon, v)
end
