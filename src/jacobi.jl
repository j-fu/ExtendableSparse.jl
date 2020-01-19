struct JacobiPreconditioner{Tv, Ti} <: AbstractExtendablePreconditioner{Tv,Ti}
    extmatrix::ExtendableSparseMatrix{Tv,Ti}
    invdiag::Array{Tv,1}
end

function update!(precon::JacobiPreconditioner)
    cscmatrix=precon.extmatrix.cscmatrix
    invdiag=precon.invdiag
    n=cscmatrix.n
    @inbounds for i=1:n
        invdiag[i]=1.0/cscmatrix[i,i]
    end
    precon
end

function JacobiPreconditioner(extmatrix::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @assert size(extmatrix,1)==size(extmatrix,2)
    flush!(extmatrix)
    invdiag=Array{Tv,1}(undef,extmatrix.cscmatrix.n)
    precon=JacobiPreconditioner{Tv, Ti}(extmatrix,invdiag)
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
