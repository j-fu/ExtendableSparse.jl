"""
$(TYPEDEF)

LU Factorization
"""
mutable struct AMGPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    amg
    phash::UInt64
end

function AMGPreconditioner(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @inbounds flush!(A)
    p=AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(A.cscmatrix))
    AMGPreconditioner{Tv,Ti}(p,A.phash)
end

function factorize!(precon::AMGPreconditioner{Tv,Ti}, A::ExtendableSparseMatrix{Tv,Ti}; kwargs...) where {Tv,Ti}
    @inbounds flush!(A)
    p=AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(A.cscmatrix))
    AMGPreconditioner{Tv,Ti}(p,A.phash)
end

LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, precon::AMGPreconditioner, v::AbstractArray{T,1} where T) = ldiv!(u,precon.amg,v)

LinearAlgebra.ldiv!(precon::AMGPreconditioner, v::AbstractArray{T,1} where T)=ldiv!(precon.amg,v)
