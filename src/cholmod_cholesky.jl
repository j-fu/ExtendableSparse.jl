mutable struct CholeskyFactorization{Tv, Ti} <: AbstractExtendableSparseLU{Tv,Ti}
    A::Union{ExtendableSparseMatrix{Tv,Ti},Nothing}
    fact::Union{SuiteSparse.CHOLMOD.Factor{Tv},Nothing}
    phash::UInt64
    A64
end

"""
$(SIGNATURES)

Default Cholesky factorization via cholmod.
"""
CholeskyFactorization()=CholeskyFactorization{Float64,Int64}(nothing,nothing,0,nothing)


function CholeskyFactorization(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    flush!(A)
    A64=Symmetric(SparseMatrixCSC{Float64,Int64}(A.cscmatrix))
    CholeskyFactorization(A,cholesky(A64),A.phash,A64)
end

function update!(cholfact::CholeskyFactorization)
    A=cholfact.A
    flush!(A)
    if A.phash!=cholfact.phash
        cholfact.A64=Symmetric(SparseMatrixCSC{Float64,Int64}(A.cscmatrix))
        cholfact.fact=cholesky(cholfact.A64)
        cholfact.phash=A.phash
    else
        cholfact.A64.data.nzval.=A.cscmatrix.nzval
        cholfact.fact=cholesky!(cholfact.fact,cholfact.A64)
    end
    cholfact
end


LinearAlgebra.ldiv!(fact::CholeskyFactorization, v)=fact.fact\v
LinearAlgebra.ldiv!(u,fact::CholeskyFactorization, v)=u.=fact.fact\v

