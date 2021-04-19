mutable struct LUFactorization{Tv,Ti} <: AbstractExtendableSparseLU{Tv,Ti}
    A::Union{Nothing,ExtendableSparseMatrix{Tv,Ti}}
    fact::Union{Nothing,SuiteSparse.UMFPACK.UmfpackLU{Tv,Ti}}
    phash::UInt64
end


"""
$(SIGNATURES)
        
Default Julia LU Factorization based on umfpack.
"""
LUFactorization()=LUFactorization{Float64,Int64}(nothing,nothing,0)

LUFactorization(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti} = factorize!(LUFactorization(),A)

function update!(lufact::LUFactorization)
    flush!(lufact.A)
    if lufact.A.phash!=lufact.phash
        lufact.fact=lu(lufact.A.cscmatrix)
        lufact.phash=lufact.A.phash
    else
        lufact.fact=lu!(lufact.fact,lufact.A.cscmatrix)
    end
    lufact
end

