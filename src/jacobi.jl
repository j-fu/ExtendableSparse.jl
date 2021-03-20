"""
$(TYPEDEF)

Jacobi preconditoner
"""
struct JacobiPreconditioner{Tv, Ti} <: AbstractExtendablePreconditioner{Tv,Ti}
    extmatrix::ExtendableSparseMatrix{Tv,Ti}
    invdiag::Array{Tv,1}
end

"""
$(SIGNATURES)

Update Jacobi preconditoner
"""
function update!(precon::JacobiPreconditioner)
    cscmatrix=precon.extmatrix.cscmatrix
    invdiag=precon.invdiag
    n=cscmatrix.n
    @inbounds for i=1:n
        invdiag[i]=1.0/cscmatrix[i,i]
    end
    precon
end

"""
$(SIGNATURES)

Construct Jacobi preconditoner
"""
function JacobiPreconditioner(extmatrix::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @assert size(extmatrix,1)==size(extmatrix,2)
    flush!(extmatrix)
    invdiag=Array{Tv,1}(undef,extmatrix.cscmatrix.n)
    precon=JacobiPreconditioner{Tv, Ti}(extmatrix,invdiag)
    update!(precon)
end

"""
$(SIGNATURES)

Construct Jacobi preconditoner
"""
JacobiPreconditioner(cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}=JacobiPreconditioner(ExtendableSparseMatrix(cscmatrix))


"""
$(SIGNATURES)

Solve  Jacobi preconditioning system
"""
function  LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, precon::JacobiPreconditioner, v::AbstractArray{T,1} where T)
    invdiag=precon.invdiag
    n=length(invdiag)
    @inbounds @simd for i=1:n
        u[i]=invdiag[i]*v[i]
    end
end

"""
$(SIGNATURES)

Inplace solve  Jacobi preconditioning system
"""
function LinearAlgebra.ldiv!(precon::JacobiPreconditioner, v::AbstractArray{T,1} where T)
    ldiv!(v, precon, v)
end
