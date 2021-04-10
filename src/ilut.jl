"""
$(TYPEDEF)

LU Factorization
"""
mutable struct ILUTPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    ilu::IncompleteLU.ILUFactorization{Tv,Ti}
    droptol::Float64
    phash::UInt64
end

function ILUTPreconditioner(A::ExtendableSparseMatrix; droptol=1.0e-3)
    @inbounds flush!(A)
    ILUTPreconditioner(IncompleteLU.ilu(A.cscmatrix,τ=droptol),droptol,A.phash)
end

function factorize!(precon::ILUTPreconditioner, A::ExtendableSparseMatrix; kwargs...)
    @inbounds flush!(A)
    ILUTPreconditioner(IncompleteLU.ilu(A.cscmatrix,τ=precon.droptol),precon.droptol,A.phash)
end

LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, precon::ILUTPreconditioner, v::AbstractArray{T,1} where T) = ldiv!(u,precon.ilu,v)

LinearAlgebra.ldiv!(precon::ILUTPreconditioner, v::AbstractArray{T,1} where T)=ldiv!(precon.ilu,v)
